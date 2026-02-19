import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .utils import now_ts


class LatestFrameGrabber:
    """
    Captura RTSP en un thread y guarda SIEMPRE el último frame reescalado.
    Evita buffering y baja la latencia.
    """
    def __init__(self, rtsp: str, process_w: int, backoff: float, capture_fps_limit: float):
        self.rtsp = rtsp
        self.process_w = process_w
        self.backoff = backoff
        self.capture_fps_limit = max(1.0, capture_fps_limit)

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_ts: float = 0.0
        self._orig_shape: Optional[Tuple[int, int]] = None  # (h,w)
        self._proc_shape: Optional[Tuple[int, int]] = None  # (h,w)

        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thr.start()

    def stop(self):
        self._stop.set()
        self._thr.join(timeout=2.0)

    def get_latest(self):
        with self._lock:
            if self._frame is None:
                return None, 0.0, self._orig_shape, self._proc_shape
            return self._frame.copy(), self._frame_ts, self._orig_shape, self._proc_shape

    def _run(self):
        cap = None
        next_t = 0.0

        while not self._stop.is_set():
            if cap is None or not cap.isOpened():
                print("[grab] conectando rtsp...")
                cap = cv2.VideoCapture(self.rtsp)
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # no siempre aplica, pero ayuda
                except Exception:
                    pass

                if not cap.isOpened():
                    print(f"[grab] fallo conexión, reintento en {self.backoff}s")
                    time.sleep(self.backoff)
                    continue

            ok, frame = cap.read()
            if not ok or frame is None:
                print("[grab] lectura falló, reconectando...")
                cap.release()
                cap = None
                time.sleep(self.backoff)
                continue

            t = now_ts()
            if t < next_t:
                time.sleep(max(0.0, next_t - t))
            next_t = now_ts() + (1.0 / self.capture_fps_limit)

            h, w = frame.shape[:2]
            scale = self.process_w / float(w)
            frame_p = cv2.resize(frame, (self.process_w, int(h * scale)), interpolation=cv2.INTER_AREA)

            with self._lock:
                self._frame = frame_p
                self._frame_ts = now_ts()
                self._orig_shape = (h, w)
                self._proc_shape = (frame_p.shape[0], frame_p.shape[1])