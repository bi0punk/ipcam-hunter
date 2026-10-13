import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from .cfg import load_cfg, pick_rtsp, resolve_events_dir
from .dwell import evaluate_dwell, inside_stats
from .events import (
    EventContext,
    EventWorker,
    build_overlay_payload,
    write_overlay_json,
    write_status_json,
)
from .roi_json import load_roi_from_json
from .rtsp import LatestFrameGrabber
from .tracker import IoUTracker
from .types import BBox
from .utils import iso_time, now_ts
from .waha import WahaClient


def _detect(model, frame_p: np.ndarray, conf_thres: float, detect_every: int, frame_idx: int) -> list[BBox]:
    """Detalle YOLO cada `detect_every` frames; devuelve bboxes de personas."""
    if frame_idx % detect_every != 0:
        return []
    res = model.predict(frame_p, conf=conf_thres, classes=[0], verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []
    xyxy = res.boxes.xyxy.cpu().numpy()
    return [(float(x1), float(y1), float(x2), float(y2)) for x1, y1, x2, y2 in xyxy]


def _infer_count(infer_count: int, detect_every: int, frame_idx: int) -> int:
    """Conserva la semántica original: un intento de inferencia por frame elegible."""
    if frame_idx % detect_every == 0:
        return infer_count + 1
    return infer_count


def _make_job(stamp: str, ts_iso: str, roi_name: str, track, dwell: float, frame_p, tracks, dwell_map) -> dict:
    return {
        "stamp": stamp,
        "ts_iso": ts_iso,
        "roi": roi_name,
        "tid": track.tid,
        "dwell": dwell,
        "frame_p": frame_p,
        "tracks": tracks,
        "dwell_map": dwell_map,
    }


def _wait_first_frame(grabber, timeout_s: float = 10.0) -> tuple[np.ndarray | None, tuple[int, int] | None]:
    """Espera el primer frame para conocer el shape de proceso (para ROI json)."""
    t0 = now_ts()
    frame_p: np.ndarray | None = None
    proc_shape: tuple[int, int] | None = None
    while proc_shape is None and (now_ts() - t0) < timeout_s:
        frame_p, _, _, proc_shape = grabber.get_latest()
        time.sleep(0.1)
    return frame_p, proc_shape


def _resolve_roi(cfg: dict, proc_shape: tuple[int, int], default_points: np.ndarray) -> np.ndarray:
    roi_cfg = cfg.get("roi", {}) or {}
    if not bool(roi_cfg.get("from_json", False)):
        return default_points
    json_path = Path(str(roi_cfg.get("json_path", "roi.json")))
    try:
        pts = load_roi_from_json(json_path, target_proc_shape_hw=proc_shape)
        print(f"[roi] cargado desde {json_path} -> points={pts.tolist()}")
        return pts
    except Exception as e:
        print(f"[roi][warn] no pude cargar roi.json ({json_path}): {e}. Usando roi.points del config.")
        return default_points


def build_status_payload(
    ts_iso: str,
    roi_name: str,
    tracks,
    dwell_map: dict[int, float],
    cooldown_sec: float,
    last_event_name: str | None,
    chat_id: str,
    infer_fps: float,
    process_w: int,
    events_dir: Path,
    cfg_path: Path,
    grab_frame_age_ms: int,
) -> dict[str, Any]:
    inside_count, max_dwell = inside_stats(tracks, dwell_map)
    return {
        "ts": ts_iso,
        "roi": roi_name,
        "inside_count": inside_count,
        "max_dwell_s": round(max_dwell, 2),
        "cooldown_s": cooldown_sec,
        "last_event": last_event_name,
        "chat_id": chat_id,
        "infer_fps": round(infer_fps, 2),
        "process_width": process_w,
        "events_dir": str(events_dir),
        "config_path": str(cfg_path),
        "grab_frame_age_ms": grab_frame_age_ms,
    }


def run(cfg_path: Path) -> None:
    if not cfg_path.exists():
        raise SystemExit(f"Config no existe: {cfg_path}")

    cfg = load_cfg(cfg_path)

    rtsp = pick_rtsp(cfg)
    if not rtsp:
        raise SystemExit("Falta RTSP. Define CAM_RTSP_URL (y opcional CAM_RTSP_URL_SUB)")

    camera = cfg["camera"]
    process_w = int(camera["process_width"])
    fps_limit = float(camera["fps_limit"])
    capture_fps_limit = float(camera.get("capture_fps_limit", max(12.0, fps_limit)))
    backoff = float(camera["reconnect_backoff_sec"])
    detect_every = int(camera.get("detect_every_n_frames", 1))
    overlay_hz = float(camera.get("overlay_hz", 10))

    weights = cfg["model"]["yolo_weights"]
    conf_thres = float(cfg["model"]["conf_thres"])

    roi_name = cfg["roi"].get("name", "roi")
    roi_points = np.array(cfg["roi"]["points"], dtype=np.int32)

    dwell_sec = float(cfg["logic"]["dwell_seconds"])
    cooldown_sec = float(cfg["logic"]["cooldown_seconds"])
    match_thres = float(cfg["logic"]["iou_match_thres"])
    ttl_sec = float(cfg["logic"]["track_ttl_seconds"])

    events_dir = resolve_events_dir(str(cfg["storage"]["events_dir"]))
    events_dir.mkdir(parents=True, exist_ok=True)
    save_annotated = bool(cfg["storage"].get("save_annotated", True))

    waha_base = os.getenv("WAHA_URL", "").strip() or (cfg.get("whatsapp", {}) or {}).get("waha_url", "http://waha:3000")
    waha_key = os.getenv("WAHA_API_KEY", "").strip() or (cfg.get("whatsapp", {}) or {}).get("api_key", "")
    waha_session = os.getenv("WAHA_SESSION", "").strip() or (cfg.get("whatsapp", {}) or {}).get("session", "default")
    chat_id = os.getenv("WAHA_CHAT_ID", "").strip() or (cfg.get("whatsapp", {}) or {}).get("chat_id", "")

    if not chat_id:
        raise SystemExit("Falta chat_id. Define WAHA_CHAT_ID o whatsapp.chat_id en config.yaml")

    caption_tpl = cfg["whatsapp"]["caption_template"]
    waha = WahaClient(waha_base, waha_session, waha_key)

    evidence_cfg = cfg.get("evidence", {}) or {}
    evidence_mode = str(evidence_cfg.get("mode", "rtsp")).strip().lower()
    evidence_timeout_s = float(evidence_cfg.get("timeout_s", 4.0))
    fallback_to_rtsp = bool(evidence_cfg.get("fallback_to_rtsp", True))

    isapi_cfg = cfg.get("isapi", {}) or {}
    isapi_url = os.getenv("ISAPI_URL", "").strip() or str(isapi_cfg.get("url", "")).strip()
    isapi_auth = os.getenv("ISAPI_AUTH", "").strip().lower() or str(isapi_cfg.get("auth", "digest")).strip().lower()
    isapi_user = os.environ.get("ISAPI_USER", "").strip()
    isapi_pass = os.environ.get("ISAPI_PASS", "").strip()

    if evidence_mode == "isapi" and (not isapi_url or not isapi_user or not isapi_pass):
        print("[warn] evidence.mode=isapi pero faltan ISAPI_URL/USER/PASS -> usando rtsp fallback")
        evidence_mode = "rtsp"

    model = YOLO(weights)
    tracker = IoUTracker(match_thres, ttl_sec)

    rtsp_safe = rtsp.split("@")[-1] if "@" in rtsp else rtsp
    print(f"[start] cfg={cfg_path} | events_dir={events_dir} | rtsp={rtsp_safe} | process_w={process_w} | conf={conf_thres}")

    grabber = LatestFrameGrabber(rtsp, process_w, backoff, capture_fps_limit)
    grabber.start()

    _frame0, proc_shape = _wait_first_frame(grabber)
    if proc_shape is None:
        raise SystemExit("No pude obtener primer frame desde RTSP (10s). Revisa RTSP/credenciales/red.")

    roi_points = _resolve_roi(cfg, proc_shape, roi_points)
    print(f"[roi] name={roi_name} points={roi_points.tolist()} proc_shape(h,w)={proc_shape}")

    ctx = EventContext(
        events_dir=events_dir,
        evidence_mode=evidence_mode,
        isapi_url=isapi_url,
        isapi_user=isapi_user,
        isapi_pass=isapi_pass,
        isapi_auth=isapi_auth,
        evidence_timeout_s=evidence_timeout_s,
        fallback_to_rtsp=fallback_to_rtsp,
        save_annotated=save_annotated,
        caption_template=caption_tpl,
        roi_points=roi_points,
        chat_id=chat_id,
        waha=waha,
    )
    worker = EventWorker(ctx, maxsize=10)
    worker.start()

    last_alert_ts = 0.0
    last_status_write = 0.0
    last_overlay_write = 0.0
    last_event_name = None
    frame_idx = 0

    infer_count = 0
    infer_t0 = now_ts()
    grab_last_ts = 0.0

    while True:
        frame_p, fts, orig_shape, proc_shape = grabber.get_latest()
        if frame_p is None:
            time.sleep(0.05)
            continue

        grab_last_ts = fts
        ts = now_ts()

        dets = _detect(model, frame_p, conf_thres, detect_every, frame_idx)
        infer_count = _infer_count(infer_count, detect_every, frame_idx)

        tracks = tracker.update(dets, ts)

        result = evaluate_dwell(tracks, roi_points, ts, dwell_sec, cooldown_sec, last_alert_ts)
        last_alert_ts = result.new_alert_ts

        if result.triggered_track is not None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            job = _make_job(stamp, iso_time(), roi_name, result.triggered_track, result.triggered_dwell,
                            frame_p, tracks, result.dwell_map)
            if worker.submit(job):
                last_event_name = f"{stamp}_tid{result.triggered_track.tid}_{roi_name}.jpg"

        if (ts - last_overlay_write) >= (1.0 / max(1.0, overlay_hz)):
            overlay_payload = build_overlay_payload(
                iso_time(), roi_name, roi_points, orig_shape, proc_shape, tracks, result.dwell_map
            )
            write_overlay_json(events_dir / "overlay.json", overlay_payload)
            last_overlay_write = ts

        if (ts - last_status_write) >= 1.0:
            dt = max(1e-6, ts - infer_t0)
            infer_fps = infer_count / dt

            payload = build_status_payload(
                ts_iso=iso_time(),
                roi_name=roi_name,
                tracks=tracks,
                dwell_map=result.dwell_map,
                cooldown_sec=cooldown_sec,
                last_event_name=last_event_name,
                chat_id=chat_id,
                infer_fps=infer_fps,
                process_w=process_w,
                events_dir=events_dir,
                cfg_path=cfg_path,
                grab_frame_age_ms=int(max(0.0, (now_ts() - grab_last_ts)) * 1000),
            )
            write_status_json(events_dir / "status.json", payload)

            last_status_write = ts
            infer_count = 0
            infer_t0 = ts

        time.sleep(max(0.0, (1.0 / max(1e-6, fps_limit))))
        frame_idx += 1
