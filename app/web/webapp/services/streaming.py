"""Helpers de framing MJPEG (multipart/x-mixed-replace) testeables."""
import time
from collections.abc import Callable, Iterator


def mjpeg_chunk(jpg: bytes, boundary: bytes = b"--frame\r\n") -> bytes:
    """Empaqueta un JPEG como parte de un stream multipart."""
    headers = (
        boundary
        + b"Content-Type: image/jpeg\r\n"
        + f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
    )
    return headers + jpg + b"\r\n"


def mjpeg_frames(
    get_frame: Callable[[], tuple[bytes | None, object]],
    is_stopped: Callable[[], bool],
    poll_s: float = 0.01,
    idle_sleep: float = 0.2,
) -> Iterator[bytes]:
    """Genera chunks MJPEG hasta que `is_stopped()` sea True.

    Si aún no hay frame, espera `idle_sleep`; si lo hay, emite y duerme `poll_s`
    (la cadencia real la marca el grabber).
    """
    while not is_stopped():
        jpg, _meta = get_frame()
        if not jpg:
            time.sleep(idle_sleep)
            continue
        yield mjpeg_chunk(jpg)
        time.sleep(poll_s)
