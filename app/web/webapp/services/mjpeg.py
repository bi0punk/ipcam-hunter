import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _extract_roi_poly(overlay: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Soporta varios formatos posibles:
    - overlay["roi"]["points"] = [[x,y],...]
    - overlay["roi_points"] = [[x,y],...]
    - overlay["points"] = [[x,y],...]
    """
    pts = None
    if isinstance(overlay.get("roi"), dict):
        pts = overlay["roi"].get("points")
    if pts is None:
        pts = overlay.get("roi_points")
    if pts is None:
        pts = overlay.get("points")

    if not pts or not isinstance(pts, list) or len(pts) < 3:
        return None
    try:
        arr = np.array([[int(p[0]), int(p[1])] for p in pts], dtype=np.int32)
        return arr
    except Exception:
        return None


def _extract_roi_rect(overlay: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    """
    Soporta:
    - roi_xyxy: {x1,y1,x2,y2}
    - roi_xywh: {x,y,w,h}
    """
    r = overlay.get("roi_xyxy")
    if isinstance(r, dict) and all(k in r for k in ("x1", "y1", "x2", "y2")):
        return int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
    r = overlay.get("roi_xywh")
    if isinstance(r, dict) and all(k in r for k in ("x", "y", "w", "h")):
        x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
        return x, y, x + w, y + h
    return None


@dataclass
class FrameMeta:
    ts_monotonic: float = 0.0
    grab_ms: int = 0
    ok: bool = False
    err: str = ""


class FrameHub:
    """
    1 RTSP -> N clientes MJPEG.
    - Thread captura frames a PREVIEW_FPS
    - Guarda el último JPEG + meta
    - /stream/mjpeg solo “empaqueta” y envía
    """

    def __init__(
        self,
        rtsp_url: str,
        preview_width: int = 640,
        preview_fps: int = 12,
        jpeg_quality: int = 80,
        overlay_json: str = "/data/events/overlay.json",
        status_json: str = "/data/events/status.json",
        reconnect_backoff: float = 2.0,
        stale_ms_warn: int = 1500,
    ):
        self.rtsp_url = (rtsp_url or "").strip()
        self.preview_width = int(preview_width)
        self.preview_fps = int(preview_fps)
        self.jpeg_quality = int(jpeg_quality)
        self.overlay_path = Path(overlay_json)
        self.status_path = Path(status_json)
        self.reconnect_backoff = float(reconnect_backoff)
        self.stale_ms_warn = int(stale_ms_warn)

        self._lock = threading.Lock()
        self._last_jpeg: Optional[bytes] = None
        self._meta = FrameMeta()
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thr and self._thr.is_alive():
            return
        self._stop.clear()
        self._thr = threading.Thread(target=self._run, name="framehub", daemon=True)
        self._thr.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=2.0)

    def get(self) -> Tuple[Optional[bytes], FrameMeta]:
        with self._lock:
            return self._last_jpeg, FrameMeta(**self._meta.__dict__)

    def _overlay(self, frame: np.ndarray) -> np.ndarray:
        overlay = _read_json(self.overlay_path)
        status = _read_json(self.status_path)

        out = frame

        # ROI (polígono o rect)
        poly = _extract_roi_poly(overlay)
        rect = _extract_roi_rect(overlay)

        # Si overlay trae "process_width" y tu preview_width coincide, dibuja 1:1.
        # Si NO coincide, escala por ancho como aproximación.
        base_w = None
        for k in ("process_width", "process_w", "width"):
            if k in status:
                base_w = status.get(k)
                break
            if k in overlay:
                base_w = overlay.get(k)
                break
        try:
            base_w = int(base_w) if base_w else self.preview_width
        except Exception:
            base_w = self.preview_width

        scale = float(self.preview_width) / float(base_w) if base_w > 0 else 1.0

        if poly is not None:
            pts = (poly.astype(np.float32) * scale).astype(np.int32)
            cv2.polylines(out, [pts], True, (0, 255, 0), 2)

        if rect is not None and poly is None:
            x1, y1, x2, y2 = rect
            x1, y1, x2, y2 = [int(v * scale) for v in (x1, y1, x2, y2)]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Stats (status.json)
        ts = status.get("ts") or ""
        inside = status.get("inside_count")
        md = status.get("max_dwell_s")
        fps = status.get("infer_fps")

        lines = []
        if ts:
            lines.append(f"ts: {ts}")
        if inside is not None:
            lines.append(f"inside: {inside}")
        if md is not None:
            lines.append(f"max_dwell_s: {md}")
        if fps is not None:
            lines.append(f"infer_fps: {fps}")

        # Estado de stream (stale)
        age_ms = int((time.monotonic() - self._meta.ts_monotonic) * 1000.0) if self._meta.ts_monotonic else 999999
        lines.append(f"grab_age_ms: {age_ms}")
        if age_ms > self.stale_ms_warn:
            # borde rojo si está “viejo”
            h, w = out.shape[:2]
            cv2.rectangle(out, (2, 2), (w - 3, h - 3), (0, 0, 255), 6)

        # Render texto
        y = 22
        for s in lines:
            cv2.putText(out, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y += 20

        return out

    def _encode_jpeg(self, frame: np.ndarray) -> Optional[bytes]:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return None
        return buf.tobytes()

    def _run(self) -> None:
        if not self.rtsp_url:
            with self._lock:
                self._meta = FrameMeta(ok=False, err="CAM_RTSP_URL_PREVIEW vacío")
            return

        cap = None
        next_t = 0.0

        while not self._stop.is_set():
            try:
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(self.rtsp_url)
                    if not cap.isOpened():
                        with self._lock:
                            self._meta = FrameMeta(ok=False, err="No pude abrir RTSP (cap.isOpened=false)")
                        time.sleep(self.reconnect_backoff)
                        continue

                t0 = time.monotonic()
                ok, frame = cap.read()
                grab_ms = int((time.monotonic() - t0) * 1000.0)

                if not ok or frame is None:
                    with self._lock:
                        self._meta = FrameMeta(ok=False, err="cap.read() falló", grab_ms=grab_ms, ts_monotonic=time.monotonic())
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                    time.sleep(self.reconnect_backoff)
                    continue

                # resize a preview_width (mantiene aspecto)
                h, w = frame.shape[:2]
                if w > 0 and self.preview_width > 0 and w != self.preview_width:
                    scale = self.preview_width / float(w)
                    frame = cv2.resize(frame, (self.preview_width, int(h * scale)), interpolation=cv2.INTER_AREA)

                # overlay ROI + stats
                frame = self._overlay(frame)

                jpg = self._encode_jpeg(frame)
                if jpg:
                    with self._lock:
                        self._last_jpeg = jpg
                        self._meta = FrameMeta(ok=True, err="", grab_ms=grab_ms, ts_monotonic=time.monotonic())

                # throttle
                if self.preview_fps > 0:
                    next_t = max(next_t, time.monotonic()) + (1.0 / float(self.preview_fps))
                    dt = next_t - time.monotonic()
                    if dt > 0:
                        time.sleep(dt)

            except Exception as e:
                with self._lock:
                    self._meta = FrameMeta(ok=False, err=f"exception: {e}", ts_monotonic=time.monotonic())
                try:
                    if cap:
                        cap.release()
                except Exception:
                    pass
                cap = None
                time.sleep(self.reconnect_backoff)

        try:
            if cap:
                cap.release()
        except Exception:
            pass

    def mjpeg_iter(self):
        boundary = b"--frame\r\n"
        while True:
            if self._stop.is_set():
                break
            jpg, meta = self.get()
            if not jpg:
                # si aún no hay frame, espera un poco
                time.sleep(0.2)
                continue
            headers = (
                boundary
                + b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
            )
            yield headers + jpg + b"\r\n"
            # la cadencia real la marca el grabber; aquí solo evitamos loop caliente
            time.sleep(0.01)