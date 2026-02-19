import json
import os
import time
import threading
import queue
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from ultralytics import YOLO

from .cfg import load_cfg, resolve_events_dir, pick_rtsp
from .evidence import fetch_isapi_snapshot
from .roi_json import load_roi_from_json
from .rtsp import LatestFrameGrabber
from .tracker import IoUTracker
from .types import BBox
from .utils import now_ts, iso_time, safe_write_json
from .vision import centroid, point_in_roi, draw_overlay
from .waha import WahaClient


def run(cfg_path: Path) -> None:
    if not cfg_path.exists():
        raise SystemExit(f"Config no existe: {cfg_path}")

    cfg = load_cfg(cfg_path)

    rtsp = pick_rtsp(cfg)
    if not rtsp:
        raise SystemExit("Falta RTSP. Define CAM_RTSP_URL (y opcional CAM_RTSP_URL_SUB)")

    process_w = int(cfg["camera"]["process_width"])
    fps_limit = float(cfg["camera"]["fps_limit"])
    capture_fps_limit = float(cfg["camera"].get("capture_fps_limit", max(12.0, fps_limit)))
    backoff = float(cfg["camera"]["reconnect_backoff_sec"])
    detect_every = int(cfg["camera"].get("detect_every_n_frames", 1))
    overlay_hz = float(cfg["camera"].get("overlay_hz", 10))

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

    # Esperar primer frame para poder cargar ROI desde roi.json (opcional)
    frame0, _, _, proc_shape = None, 0.0, None, None
    t0 = now_ts()
    while proc_shape is None and (now_ts() - t0) < 10.0:
        frame0, _, _, proc_shape = grabber.get_latest()
        time.sleep(0.1)

    if proc_shape is None:
        raise SystemExit("No pude obtener primer frame desde RTSP (10s). Revisa RTSP/credenciales/red.")

    roi_cfg = cfg.get("roi", {}) or {}
    if bool(roi_cfg.get("from_json", False)):
        json_path = Path(str(roi_cfg.get("json_path", "roi.json")))
        try:
            roi_points = load_roi_from_json(json_path, target_proc_shape_hw=proc_shape)
            print(f"[roi] cargado desde {json_path} -> points={roi_points.tolist()}")
        except Exception as e:
            print(f"[roi][warn] no pude cargar roi.json ({json_path}): {e}. Usando roi.points del config.")

    print(f"[roi] name={roi_name} points={roi_points.tolist()} proc_shape(h,w)={proc_shape}")

    # worker async para eventos
    event_q: "queue.Queue[dict]" = queue.Queue(maxsize=10)

    def event_worker():
        while True:
            job = event_q.get()
            if job is None:
                return
            try:
                ts_iso = job["ts_iso"]
                stamp = job["stamp"]
                roi = job["roi"]
                tid = job["tid"]
                dwell = job["dwell"]
                frame_p = job["frame_p"]
                tracks = job["tracks"]
                dwell_map = job["dwell_map"]

                base = f"{stamp}_tid{tid}_{roi}"
                jpg_path = events_dir / f"{base}.jpg"
                annot_path = events_dir / f"{base}_annot.jpg"

                saved_ok = False
                skip_send = False

                if evidence_mode == "isapi":
                    try:
                        data = fetch_isapi_snapshot(isapi_url, isapi_user, isapi_pass, isapi_auth, evidence_timeout_s)
                        jpg_path.write_bytes(data)
                        saved_ok = True
                        print(f"[evidence] isapi ok -> {jpg_path.name} ({len(data)} bytes)")
                    except Exception as e:
                        print(f"[evidence][warn] isapi fallo: {e}")
                        if not fallback_to_rtsp:
                            skip_send = True

                if (not saved_ok) and (not skip_send):
                    out_img = frame_p
                    if save_annotated:
                        out_img = draw_overlay(frame_p, roi_points, tracks, dwell_map, iso_time)
                    cv2.imwrite(str(jpg_path), out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

                if save_annotated:
                    try:
                        over = draw_overlay(frame_p, roi_points, tracks, dwell_map, iso_time)
                        cv2.imwrite(str(annot_path), over, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    except Exception as e:
                        print(f"[annot][warn] no pude guardar overlay: {e}")

                caption = caption_tpl.format(roi=roi, dwell=dwell, ts=ts_iso)

                if skip_send:
                    print("[alert][warn] no evidencia y fallback_to_rtsp=false -> no envío")
                else:
                    try:
                        resp = waha.send_image_b64(chat_id, jpg_path, caption)
                        print(f"[alert] enviado {jpg_path.name} -> {chat_id} | resp={json.dumps(resp)[:200]}")
                    except Exception as e:
                        print(f"[alert][error] fallo envío WhatsApp: {e}")

            except Exception as e:
                print(f"[worker][error] {e}")
            finally:
                event_q.task_done()

    threading.Thread(target=event_worker, daemon=True).start()

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

        dets: List[BBox] = []
        if frame_idx % detect_every == 0:
            res = model.predict(frame_p, conf=conf_thres, classes=[0], verbose=False)[0]
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                for x1, y1, x2, y2 in xyxy:
                    dets.append((float(x1), float(y1), float(x2), float(y2)))
            infer_count += 1

        tracks = tracker.update(dets, ts)

        dwell_map: Dict[int, float] = {}
        triggered = False
        triggered_track = None
        triggered_dwell = 0.0

        for tr in tracks:
            c = centroid(tr.bbox)
            inside = point_in_roi(c, roi_points)

            if inside:
                if tr.in_roi_since is None:
                    tr.in_roi_since = ts
                    tr.fired = False
                dwell = ts - tr.in_roi_since
                dwell_map[tr.tid] = dwell

                if (not tr.fired) and (dwell >= dwell_sec):
                    if (ts - last_alert_ts) >= cooldown_sec:
                        triggered = True
                        triggered_track = tr
                        triggered_dwell = dwell
                        tr.fired = True
                        last_alert_ts = ts
            else:
                tr.in_roi_since = None
                tr.fired = False

        if triggered and triggered_track is not None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            job = {
                "stamp": stamp,
                "ts_iso": iso_time(),
                "roi": roi_name,
                "tid": triggered_track.tid,
                "dwell": triggered_dwell,
                "frame_p": frame_p,
                "tracks": tracks,
                "dwell_map": dwell_map
            }
            try:
                event_q.put_nowait(job)
                last_event_name = f"{stamp}_tid{triggered_track.tid}_{roi_name}.jpg"
            except queue.Full:
                print("[worker][warn] cola llena, dropeo evento (evitar bloqueo)")

        # overlay.json
        if (ts - last_overlay_write) >= (1.0 / max(1.0, overlay_hz)):
            out = {
                "ts": iso_time(),
                "roi": {"name": roi_name, "points": roi_points.tolist()},
                "frame": {
                    "orig_hw": list(orig_shape) if orig_shape else None,
                    "proc_hw": list(proc_shape) if proc_shape else None,
                },
                "tracks": []
            }
            for tr in tracks:
                c = centroid(tr.bbox)
                inside = point_in_roi(c, roi_points)
                out["tracks"].append({
                    "id": tr.tid,
                    "bbox": [round(tr.bbox[0], 2), round(tr.bbox[1], 2), round(tr.bbox[2], 2), round(tr.bbox[3], 2)],
                    "centroid": [int(c[0]), int(c[1])],
                    "inside": bool(inside),
                    "dwell_s": round(dwell_map.get(tr.tid, 0.0), 2)
                })
            try:
                safe_write_json(events_dir / "overlay.json", out)
            except Exception as e:
                print(f"[overlay][error] {e}")
            last_overlay_write = ts

        # status.json
        if (ts - last_status_write) >= 1.0:
            inside_count = 0
            max_dwell = 0.0
            for tr in tracks:
                d = dwell_map.get(tr.tid, 0.0)
                if d > 0:
                    inside_count += 1
                    max_dwell = max(max_dwell, d)

            dt = max(1e-6, ts - infer_t0)
            infer_fps = infer_count / dt

            payload = {
                "ts": iso_time(),
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
                "grab_frame_age_ms": int(max(0.0, (now_ts() - grab_last_ts)) * 1000),
            }
            try:
                safe_write_json(events_dir / "status.json", payload)
            except Exception as e:
                print(f"[status][error] {e}")

            last_status_write = ts
            infer_count = 0
            infer_t0 = ts

        time.sleep(max(0.0, (1.0 / max(1e-6, fps_limit))))
        frame_idx += 1