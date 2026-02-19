#!/usr/bin/env python3
import argparse
import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth, HTTPBasicAuth
import yaml
from ultralytics import YOLO

# Soporte opcional: permite correr el script fuera de docker compose leyendo .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2


def now_ts() -> float:
    return time.monotonic()


def iso_time() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def fetch_isapi_snapshot(
    url: str,
    username: str,
    password: str,
    auth_mode: str = "digest",
    timeout_s: float = 4.0
) -> bytes:
    """Obtiene snapshot JPEG por ISAPI (Hikvision) usando Basic o Digest.

    Retorna bytes JPEG. Lanza excepción si la respuesta no parece una imagen.
    """
    auth_mode = (auth_mode or "digest").strip().lower()
    if auth_mode == "basic":
        auth = HTTPBasicAuth(username, password)
    else:
        auth = HTTPDigestAuth(username, password)

    r = requests.get(url, auth=auth, timeout=timeout_s)
    r.raise_for_status()
    data = r.content or b""

    # Validación rápida de JPEG (magic bytes)
    if not data.startswith(b"\xff\xd8"):
        head = data[:200].decode("utf-8", errors="replace")
        raise RuntimeError(f"ISAPI snapshot no es JPEG (len={len(data)}). Head: {head}")

    return data


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def centroid(bb: BBox) -> Tuple[int, int]:
    x1, y1, x2, y2 = bb
    return (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))


@dataclass
class Track:
    tid: int
    bbox: BBox
    last_seen: float
    in_roi_since: Optional[float] = None
    fired: bool = False


class IoUTracker:
    """Tracker minimalista por IoU (suficiente para 1 cámara y condición 'cualquiera en ROI')."""
    def __init__(self, match_thres: float, ttl_sec: float):
        self.match_thres = match_thres
        self.ttl_sec = ttl_sec
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    def update(self, dets: List[BBox], ts: float) -> List[Track]:
        # Expirar tracks
        dead = [tid for tid, tr in self.tracks.items() if (ts - tr.last_seen) > self.ttl_sec]
        for tid in dead:
            del self.tracks[tid]

        assigned = set()

        # Match con tracks existentes
        for tid, tr in list(self.tracks.items()):
            best_iou = 0.0
            best_j = None
            for j, bb in enumerate(dets):
                if j in assigned:
                    continue
                val = iou(tr.bbox, bb)
                if val > best_iou:
                    best_iou = val
                    best_j = j
            if best_j is not None and best_iou >= self.match_thres:
                tr.bbox = dets[best_j]
                tr.last_seen = ts
                assigned.add(best_j)

        # Crear nuevos tracks
        for j, bb in enumerate(dets):
            if j in assigned:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = Track(tid=tid, bbox=bb, last_seen=ts)

        return list(self.tracks.values())


class WahaClient:
    def __init__(self, base_url: str, session: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.session = session
        self.api_key = api_key.strip()

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["X-Api-Key"] = self.api_key
        return h

    def send_image_b64(self, chat_id: str, jpeg_path: Path, caption: str = "") -> dict:
        data_b64 = base64.b64encode(jpeg_path.read_bytes()).decode("ascii")
        payload = {
            "session": self.session,
            "chatId": chat_id,
            "file": {
                "mimetype": "image/jpeg",
                "filename": jpeg_path.name,
                "data": data_b64
            },
            "caption": caption
        }
        url = f"{self.base_url}/api/sendImage"
        r = requests.post(url, json=payload, headers=self._headers(), timeout=25)
        r.raise_for_status()
        return r.json() if r.content else {"status": "ok"}


def load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def point_in_roi(pt: Tuple[int, int], roi_pts: np.ndarray) -> bool:
    # roi_pts: Nx2 en coordenadas del frame procesado
    return cv2.pointPolygonTest(roi_pts, pt, False) >= 0


def draw_overlay(img: np.ndarray, roi_pts: np.ndarray, tracks: List[Track], dwell_map: Dict[int, float]) -> np.ndarray:
    out = img.copy()
    cv2.polylines(out, [roi_pts.astype(np.int32)], True, (0, 255, 0), 2)
    for tr in tracks:
        x1, y1, x2, y2 = map(int, tr.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)
        d = dwell_map.get(tr.tid, 0.0)
        cv2.putText(out, f"id={tr.tid} dwell={d:.1f}s", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(out, iso_time(), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def resolve_config_path(cli_arg: str) -> Path:
    # Prioridad: CLI --config > ENV CONFIG_PATH > default
    env_cfg = os.getenv("CONFIG_PATH", "").strip()
    if cli_arg and cli_arg != "app/config.yaml":
        return Path(cli_arg)
    if env_cfg:
        return Path(env_cfg)
    return Path(cli_arg)


def resolve_events_dir(cfg_events_dir: str) -> Path:
    # 1) Si existe ENV explícito, manda
    env_events = os.getenv("EVENTS_DIR", "").strip()
    if env_events:
        return Path(env_events)

    p = Path(cfg_events_dir)

    # 2) Heurística docker: si hay volumen /data/events y el config usa relativo, redirigir a /data/events
    # Esto evita que el dashboard "no vea nada" aunque el detector esté funcionando.
    docker_events = Path("/data/events")
    if not p.is_absolute() and docker_events.exists():
        if cfg_events_dir.strip() in ("./events", "events"):
            return docker_events

    return p


def main():
    default_cfg = os.getenv("CONFIG_PATH", "").strip() or "app/config.yaml"

    ap = argparse.ArgumentParser(description="ROI dwell alert -> snapshot -> WhatsApp (waha)")
    ap.add_argument("--config", default=default_cfg, help="Ruta a config.yaml")
    args = ap.parse_args()

    cfg_path = resolve_config_path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config no existe: {cfg_path} (tip: usa --config o CONFIG_PATH)")

    cfg = load_cfg(cfg_path)

    # -----------------------------
    # Camera (RTSP): secreto -> .env
    # -----------------------------
    rtsp = os.getenv("CAM_RTSP_URL", "").strip() or (cfg.get("camera", {}) or {}).get("rtsp_url", "")
    if not rtsp:
        raise SystemExit("Falta RTSP. Define CAM_RTSP_URL en .env (recomendado) o camera.rtsp_url en config.yaml")

    process_w = int(cfg["camera"]["process_width"])
    fps_limit = float(cfg["camera"]["fps_limit"])
    backoff = float(cfg["camera"]["reconnect_backoff_sec"])
    detect_every = int(cfg["camera"].get("detect_every_n_frames", 1))

    # -----------------------------
    # Model
    # -----------------------------
    weights = cfg["model"]["yolo_weights"]
    conf_thres = float(cfg["model"]["conf_thres"])

    # -----------------------------
    # ROI / logic
    # -----------------------------
    roi_points = np.array(cfg["roi"]["points"], dtype=np.int32)
    roi_name = cfg["roi"].get("name", "roi")

    dwell_sec = float(cfg["logic"]["dwell_seconds"])
    cooldown_sec = float(cfg["logic"]["cooldown_seconds"])
    match_thres = float(cfg["logic"]["iou_match_thres"])
    ttl_sec = float(cfg["logic"]["track_ttl_seconds"])

    # -----------------------------
    # Storage
    # -----------------------------
    events_dir_cfg = str(cfg["storage"]["events_dir"])
    events_dir = resolve_events_dir(events_dir_cfg)
    events_dir.mkdir(parents=True, exist_ok=True)
    save_annotated = bool(cfg["storage"].get("save_annotated", True))

    # -----------------------------
    # WAHA: api key secreto -> .env
    # -----------------------------
    waha_base = os.getenv("WAHA_URL", "").strip() or (cfg.get("whatsapp", {}) or {}).get("waha_url", "http://waha:3000")
    waha_key = os.getenv("WAHA_API_KEY", "").strip() or (cfg.get("whatsapp", {}) or {}).get("api_key", "")
    waha_session = os.getenv("WAHA_SESSION", "").strip() or (cfg.get("whatsapp", {}) or {}).get("session", "default")
    chat_id = os.getenv("WAHA_CHAT_ID", "").strip() or (cfg.get("whatsapp", {}) or {}).get("chat_id", "")

    if not chat_id:
        raise SystemExit("Falta chat_id. Define WAHA_CHAT_ID en .env o whatsapp.chat_id en config.yaml")

    waha = WahaClient(
        base_url=waha_base,
        session=waha_session,
        api_key=waha_key
    )
    caption_tpl = cfg["whatsapp"]["caption_template"]

    # -----------------------------
    # Evidence (ISAPI)
    # -----------------------------
    evidence_cfg = cfg.get("evidence", {}) or {}
    evidence_mode = str(evidence_cfg.get("mode", "rtsp")).strip().lower()
    evidence_timeout_s = float(evidence_cfg.get("timeout_s", 4.0))
    fallback_to_rtsp = bool(evidence_cfg.get("fallback_to_rtsp", True))

    isapi_cfg = cfg.get("isapi", {}) or {}

    isapi_url = os.getenv("ISAPI_URL", "").strip() or str(isapi_cfg.get("url", "")).strip()
    isapi_auth = os.getenv("ISAPI_AUTH", "").strip().lower() or str(isapi_cfg.get("auth", "digest")).strip().lower()
    isapi_user = os.environ.get("ISAPI_USER", "").strip()
    isapi_pass = os.environ.get("ISAPI_PASS", "").strip()

    if evidence_mode == "isapi":
        if not isapi_url or not isapi_user or not isapi_pass:
            print("[warn] evidence.mode=isapi pero faltan ISAPI_URL o ISAPI_USER/ISAPI_PASS. Usando rtsp frame como evidencia.")
            evidence_mode = "rtsp"

    model = YOLO(weights)
    tracker = IoUTracker(match_thres=match_thres, ttl_sec=ttl_sec)

    last_alert_ts = 0.0
    last_status_write = 0.0
    last_event_name = None
    frame_idx = 0

    # Debug detección
    last_any_det_ts = now_ts()
    no_det_warn_every_sec = 8.0

    # No imprimir credenciales
    rtsp_safe = rtsp.split("@")[-1] if "@" in rtsp else rtsp
    print(
        f"[start] cfg={cfg_path} | events_dir={events_dir} (cfg={events_dir_cfg}) | rtsp_host={rtsp_safe} | "
        f"process_w={process_w} | conf={conf_thres} | dwell={dwell_sec}s | cooldown={cooldown_sec}s | chat={chat_id} | roi={roi_name}"
    )
    print(f"[roi] points(processed-coords)={roi_points.tolist()}")

    cap = None
    next_frame_time = 0.0

    while True:
        if cap is None or not cap.isOpened():
            print("[rtsp] conectando...")
            cap = cv2.VideoCapture(rtsp)
            if not cap.isOpened():
                print(f"[rtsp] fallo, reintento en {backoff}s")
                time.sleep(backoff)
                continue

        ok, frame = cap.read()
        if not ok or frame is None:
            print("[rtsp] lectura falló, reconectando...")
            cap.release()
            cap = None
            time.sleep(backoff)
            continue

        # Limitar FPS efectivo (analítica)
        t = now_ts()
        if t < next_frame_time:
            time.sleep(max(0.0, next_frame_time - t))
        next_frame_time = now_ts() + (1.0 / max(1e-6, fps_limit))

        # resize a ancho fijo
        h, w = frame.shape[:2]
        scale = process_w / float(w)
        frame_p = cv2.resize(frame, (process_w, int(h * scale)), interpolation=cv2.INTER_AREA)

        ts = now_ts()
        dets: List[BBox] = []

        if frame_idx % detect_every == 0:
            # YOLO: persona en COCO = class 0
            res = model.predict(frame_p, conf=conf_thres, classes=[0], verbose=False)[0]
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                # Filtro extra defensivo (por si "classes" no aplica según versión)
                if hasattr(res.boxes, "cls") and res.boxes.cls is not None:
                    cls = res.boxes.cls.cpu().numpy().astype(int)
                    for (x1, y1, x2, y2), c in zip(xyxy, cls):
                        if c == 0:
                            dets.append((float(x1), float(y1), float(x2), float(y2)))
                else:
                    for x1, y1, x2, y2 in xyxy:
                        dets.append((float(x1), float(y1), float(x2), float(y2)))

        if dets:
            last_any_det_ts = ts
        else:
            if (ts - last_any_det_ts) >= no_det_warn_every_sec and frame_idx % int(max(1, fps_limit)) == 0:
                print(
                    f"[detect][warn] {no_det_warn_every_sec:.0f}s sin detecciones (person). "
                    f"Tip: baja model.conf_thres (ej 0.25) o sube camera.process_width (ej 960/1280). "
                    f"frame_orig={w}x{h} frame_proc={frame_p.shape[1]}x{frame_p.shape[0]}"
                )
                last_any_det_ts = ts  # evitar spam

        tracks = tracker.update(dets, ts)

        dwell_map: Dict[int, float] = {}
        triggered = False
        triggered_track = None
        triggered_dwell = 0.0

        # Regla: si sale del ROI -> reset
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
            base = f"{stamp}_tid{triggered_track.tid}_{roi_name}"
            jpg_path = events_dir / f"{base}.jpg"          # evidencia enviada a WhatsApp
            annot_path = events_dir / f"{base}_annot.jpg"  # overlay para dashboard/forense

            saved_ok = False
            skip_send = False

            # 1) Evidencia principal
            if evidence_mode == "isapi":
                try:
                    data = fetch_isapi_snapshot(
                        url=isapi_url,
                        username=isapi_user,
                        password=isapi_pass,
                        auth_mode=isapi_auth,
                        timeout_s=evidence_timeout_s,
                    )
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
                    out_img = draw_overlay(frame_p, roi_points, tracks, dwell_map)
                cv2.imwrite(str(jpg_path), out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

            # 2) Overlay adicional (siempre basado en frame procesado)
            if save_annotated:
                try:
                    over = draw_overlay(frame_p, roi_points, tracks, dwell_map)
                    cv2.imwrite(str(annot_path), over, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    last_event_name = annot_path.name
                except Exception as e:
                    print(f"[annot][warn] no pude guardar overlay: {e}")
                    last_event_name = jpg_path.name
            else:
                last_event_name = jpg_path.name

            caption = caption_tpl.format(roi=roi_name, dwell=triggered_dwell, ts=iso_time())

            if skip_send:
                print("[alert][warn] evidencia no disponible y fallback_to_rtsp=false -> no se envía WhatsApp")
            else:
                try:
                    resp = waha.send_image_b64(chat_id=chat_id, jpeg_path=jpg_path, caption=caption)
                    print(f"[alert] enviado {jpg_path.name} -> {chat_id} | resp={json.dumps(resp)[:200]}")
                except Exception as e:
                    print(f"[alert][error] fallo envío WhatsApp: {e}")

        # Emitir estado (para dashboard) ~1 vez/seg
        if (ts - last_status_write) >= 1.0:
            inside_count = 0
            max_dwell = 0.0
            for tr in tracks:
                d = dwell_map.get(tr.tid, 0.0)
                if d > 0:
                    inside_count += 1
                    if d > max_dwell:
                        max_dwell = d

            status_path = events_dir / "status.json"
            payload = {
                "ts": iso_time(),
                "roi": roi_name,
                "inside_count": inside_count,
                "max_dwell_s": round(max_dwell, 2),
                "cooldown_s": cooldown_sec,
                "last_event": last_event_name,
                "chat_id": chat_id,
                "fps_limit": fps_limit,
                "process_width": process_w,
                "events_dir": str(events_dir),
                "config_path": str(cfg_path),
            }
            try:
                safe_write_json(status_path, payload)
            except Exception as e:
                print(f"[status][error] {e}")
            last_status_write = ts

        frame_idx += 1


if __name__ == "__main__":
    main()