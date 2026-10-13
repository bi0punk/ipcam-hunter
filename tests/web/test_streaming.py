from webapp.services.streaming import mjpeg_chunk, mjpeg_frames


def test_mjpeg_chunk_formato():
    jpg = b"\xff\xd8datos-jpeg\xff\xd9"
    chunk = mjpeg_chunk(jpg)
    assert chunk.startswith(b"--frame\r\n")
    assert b"Content-Type: image/jpeg\r\n" in chunk
    assert f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii") in chunk
    assert chunk.endswith(jpg + b"\r\n")


def test_mjpeg_chunk_boundary_custom():
    chunk = mjpeg_chunk(b"x", boundary=b"--otro\r\n")
    assert chunk.startswith(b"--otro\r\n")


def test_mjpeg_frames_emite_hasta_stop():
    state = {"calls": 0, "stop": False}

    def get_frame():
        return b"jpeg", None

    def is_stopped():
        return state["stop"]

    gen = mjpeg_frames(get_frame, is_stopped, poll_s=0.0)
    first = next(gen)
    assert b"jpeg" in first
    state["stop"] = True
    rest = list(gen)
    assert rest == []


def test_mjpeg_frames_espera_si_no_hay_frame():
    state = {"calls": 0, "stop": False}

    def get_frame():
        state["calls"] += 1
        if state["calls"] >= 2:
            state["stop"] = True
        return None, None

    def is_stopped():
        return state["stop"]

    gen = mjpeg_frames(get_frame, is_stopped, idle_sleep=0.0)
    assert list(gen) == []
    assert state["calls"] >= 2
