import time
from pathlib import Path
from typing import Iterator


def mjpeg_generator(live_path: Path, fps: float, boundary: str = "frame") -> Iterator[bytes]:
    delay = 1.0 / max(0.5, float(fps))
    last_mtime = 0.0

    while True:
        if live_path.exists():
            try:
                st = live_path.stat()
                # evita leer si no cambió (reduce disco)
                if st.st_mtime != last_mtime:
                    last_mtime = st.st_mtime
                    jpg = live_path.read_bytes()

                    yield (
                        f"--{boundary}\r\n"
                        "Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpg)}\r\n\r\n"
                    ).encode("utf-8") + jpg + b"\r\n"
            except Exception:
                # si hay un write parcial, espera y reintenta
                pass

        time.sleep(delay)