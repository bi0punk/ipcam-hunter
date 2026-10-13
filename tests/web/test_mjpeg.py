import json

from webapp.services.mjpeg import FrameHub, FrameMeta


def test_frame_meta_defaults():
    m = FrameMeta()
    assert m.ok is False
    assert m.ts_monotonic == 0.0
    assert m.err == ""


def test_get_sin_frame_devuelve_none():
    hub = FrameHub(rtsp_url="")
    jpg, meta = hub.get()
    assert jpg is None
    assert isinstance(meta, FrameMeta)


def test_run_sin_rtsp_marca_error():
    hub = FrameHub(rtsp_url="")
    hub._run()  # síncrono: con url vacía sale de inmediato
    jpg, meta = hub.get()
    assert jpg is None
    assert meta.ok is False
    assert "vacío" in meta.err


def test_encode_jpeg_devuelve_bytes(sample_frame):
    hub = FrameHub(rtsp_url="rtsp://x")
    jpg = hub._encode_jpeg(sample_frame())
    assert isinstance(jpg, bytes)
    assert jpg.startswith(b"\xff\xd8")


def test_prepare_reescala_a_preview_width(sample_frame):
    hub = FrameHub(rtsp_url="rtsp://x", preview_width=40)
    out = hub._prepare(sample_frame(h=30, w=80, value=0))
    assert out.shape[1] == 40


def test_prepare_no_reescala_si_igual(sample_frame):
    hub = FrameHub(rtsp_url="rtsp://x", preview_width=96)
    out = hub._prepare(sample_frame(h=30, w=96, value=0))
    assert out.shape[1] == 96


def test_overlay_usa_json_en_disco(tmp_path, sample_frame):
    overlay = tmp_path / "overlay.json"
    status = tmp_path / "status.json"
    overlay.write_text(json.dumps({"roi": {"points": [[0, 0], [50, 0], [50, 50]]}}), encoding="utf-8")
    status.write_text(json.dumps({"ts": "t", "inside_count": 1, "process_width": 96}), encoding="utf-8")
    hub = FrameHub(rtsp_url="rtsp://x", preview_width=96, overlay_json=str(overlay), status_json=str(status))
    out = hub._overlay(sample_frame(value=0))
    assert out.any()


def test_mjpeg_iter_emite_chunk(sample_frame):
    hub = FrameHub(rtsp_url="rtsp://x")
    jpg = hub._encode_jpeg(sample_frame())
    hub._last_jpeg = jpg
    gen = hub.mjpeg_iter()
    chunk = next(gen)
    assert jpg in chunk
    hub._stop.set()
    assert list(gen) == []


def test_start_stop_no_crashea_con_url_vacia():
    hub = FrameHub(rtsp_url="")
    hub.start()
    hub.stop()
    assert hub._stop.is_set()


def test_lock_protege_ultimo_jpeg():
    hub = FrameHub(rtsp_url="rtsp://x")
    hub._last_jpeg = b"abc"
    jpg, _ = hub.get()
    assert jpg == b"abc"
