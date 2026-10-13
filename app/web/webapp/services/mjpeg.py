import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .overlay import read_json, render_overlay
from .streaming import mjpeg_frames


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
        self._last_jpeg: bytes | None = None
        self._meta = FrameMeta()
        self._stop = threading.Event()
        self._thr: threading.Thread | None = None

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

    def get(self) -> tuple[bytes | None, FrameMeta]:
        with self._lock:
            return self._last_jpeg, FrameMeta(**self._meta.__dict__)

    # ----- procesamiento de frame -----

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        """Reescala a preview_width (mantiene aspecto) y aplica overlay."""
        h, w = frame.shape[:2]
        if w > 0 and self.preview_width > 0 and w != self.preview_width:
            scale = self.preview_width / float(w)
            frame = cv2.resize(frame, (self.preview_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        return self._overlay(frame)

    def _overlay(self, frame: np.ndarray) -> np.ndarray:
        overlay = read_json(self.overlay_path)
        status = read_json(self.status_path)
        return render_overlay(
            frame,
            overlay,
            status,
            preview_width=self.preview_width,
            stale_ms_warn=self.stale_ms_warn,
            meta_ts_monotonic=self._meta.ts_monotonic,
        )

    def _encode_jpeg(self, frame: np.ndarray) -> bytes | None:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return None
        return buf.tobytes()

    # ----- captura -----

    def _release(self, cap) -> None:
        try:
            if cap:
                cap.release()
        except Exception:
            pass

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
                    self._release(cap)
                    cap = None
                    time.sleep(self.reconnect_backoff)
                    continue

                frame = self._prepare(frame)
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
                self._release(cap)
                cap = None
                time.sleep(self.reconnect_backoff)

        self._release(cap)

    def mjpeg_iter(self):
        yield from mjpeg_frames(self.get, self._stop.is_set)
