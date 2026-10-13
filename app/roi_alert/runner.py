import time
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

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
from .settings import RunnerSettings
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


def _resolve_roi(s: RunnerSettings, proc_shape: tuple[int, int]) -> np.ndarray:
    if not s.roi_from_json:
        return s.roi_points
    json_path = Path(s.roi_json_path)
    try:
        pts = load_roi_from_json(json_path, target_proc_shape_hw=proc_shape)
        print(f"[roi] cargado desde {json_path} -> points={pts.tolist()}")
        return pts
    except Exception as e:
        print(f"[roi][warn] no pude cargar roi.json ({json_path}): {e}. Usando roi.points del config.")
        return s.roi_points


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


def _build_event_context(s: RunnerSettings, roi_points: np.ndarray, waha: WahaClient) -> EventContext:
    return EventContext(
        events_dir=s.events_dir,
        evidence_mode=s.evidence_mode,
        isapi_url=s.isapi_url,
        isapi_user=s.isapi_user,
        isapi_pass=s.isapi_pass,
        isapi_auth=s.isapi_auth,
        evidence_timeout_s=s.evidence_timeout_s,
        fallback_to_rtsp=s.fallback_to_rtsp,
        save_annotated=s.save_annotated,
        caption_template=s.caption_template,
        roi_points=roi_points,
        chat_id=s.chat_id,
        waha=waha,
    )


def run(cfg_path: Path) -> None:
    if not cfg_path.exists():
        raise SystemExit(f"Config no existe: {cfg_path}")

    s = RunnerSettings.from_file(cfg_path)

    if not s.rtsp:
        raise SystemExit("Falta RTSP. Define CAM_RTSP_URL (y opcional CAM_RTSP_URL_SUB)")

    for w in s.warnings:
        print(w)

    s.events_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(s.yolo_weights)
    tracker = IoUTracker(s.match_thres, s.ttl_sec)

    rtsp_safe = s.rtsp.split("@")[-1] if "@" in s.rtsp else s.rtsp
    print(f"[start] cfg={cfg_path} | events_dir={s.events_dir} | rtsp={rtsp_safe} | process_w={s.process_w} | conf={s.conf_thres}")

    grabber = LatestFrameGrabber(s.rtsp, s.process_w, s.backoff, s.capture_fps_limit)
    grabber.start()

    _frame0, proc_shape = _wait_first_frame(grabber)
    if proc_shape is None:
        raise SystemExit("No pude obtener primer frame desde RTSP (10s). Revisa RTSP/credenciales/red.")

    roi_points = _resolve_roi(s, proc_shape)
    print(f"[roi] name={s.roi_name} points={roi_points.tolist()} proc_shape(h,w)={proc_shape}")

    waha = WahaClient(s.waha_base, s.waha_session, s.waha_key)
    worker = EventWorker(_build_event_context(s, roi_points, waha), maxsize=10)
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

        dets = _detect(model, frame_p, s.conf_thres, s.detect_every, frame_idx)
        infer_count = _infer_count(infer_count, s.detect_every, frame_idx)

        tracks = tracker.update(dets, ts)

        result = evaluate_dwell(tracks, roi_points, ts, s.dwell_sec, s.cooldown_sec, last_alert_ts)
        last_alert_ts = result.new_alert_ts

        if result.triggered_track is not None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            job = _make_job(stamp, iso_time(), s.roi_name, result.triggered_track, result.triggered_dwell,
                            frame_p, tracks, result.dwell_map)
            if worker.submit(job):
                last_event_name = f"{stamp}_tid{result.triggered_track.tid}_{s.roi_name}.jpg"

        if (ts - last_overlay_write) >= (1.0 / max(1.0, s.overlay_hz)):
            overlay_payload = build_overlay_payload(
                iso_time(), s.roi_name, roi_points, orig_shape, proc_shape, tracks, result.dwell_map
            )
            write_overlay_json(s.events_dir / "overlay.json", overlay_payload)
            last_overlay_write = ts

        if (ts - last_status_write) >= 1.0:
            dt = max(1e-6, ts - infer_t0)
            infer_fps = infer_count / dt

            payload = build_status_payload(
                ts_iso=iso_time(),
                roi_name=s.roi_name,
                tracks=tracks,
                dwell_map=result.dwell_map,
                cooldown_sec=s.cooldown_sec,
                last_event_name=last_event_name,
                chat_id=s.chat_id,
                infer_fps=infer_fps,
                process_w=s.process_w,
                events_dir=s.events_dir,
                cfg_path=cfg_path,
                grab_frame_age_ms=int(max(0.0, (now_ts() - grab_last_ts)) * 1000),
            )
            write_status_json(s.events_dir / "status.json", payload)

            last_status_write = ts
            infer_count = 0
            infer_t0 = ts

        time.sleep(max(0.0, (1.0 / max(1e-6, s.fps_limit))))
        frame_idx += 1
