import json
from pathlib import Path

import numpy as np

from roi_alert import events
from roi_alert.events import (
    EventContext,
    EventWorker,
    build_caption,
    build_overlay_payload,
    save_evidence,
)


def _roi():
    return np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32)


class FakeWaha:
    def __init__(self):
        self.calls = []

    def send_image_b64(self, chat_id, path, caption=""):
        self.calls.append((chat_id, Path(path).name, caption))
        return {"status": "ok"}


def _ctx(tmp_path, frame_p, **over):
    kw = dict(
        events_dir=tmp_path,
        evidence_mode="rtsp",
        isapi_url="",
        isapi_user="",
        isapi_pass="",
        isapi_auth="digest",
        evidence_timeout_s=1.0,
        fallback_to_rtsp=True,
        save_annotated=True,
        caption_template="Alerta {roi} {dwell:.1f}s {ts}",
        roi_points=_roi(),
        chat_id="grupo@g.us",
        waha=FakeWaha(),
    )
    kw.update(over)
    return EventContext(**kw)


def _job(frame_p):
    return {
        "stamp": "20260101_120000",
        "ts_iso": "2026-01-01 12:00:00",
        "roi": "zona_1",
        "tid": 7,
        "dwell": 3.2,
        "frame_p": frame_p,
        "tracks": [],
        "dwell_map": {},
    }


def test_build_caption_formatea():
    assert build_caption("A {roi} {dwell:.1f}s {ts}", "z", 2.5, "t") == "A z 2.5s t"


def test_save_evidence_rtsp_escribe_jpg_y_annot(tmp_path, sample_frame):
    frame = sample_frame(value=100)
    ctx = _ctx(tmp_path, frame)
    path = save_evidence(_job(frame), ctx)
    assert path is not None
    assert path.name == "20260101_120000_tid7_zona_1.jpg"
    assert path.exists()
    assert path.read_bytes().startswith(b"\xff\xd8")  # JPEG
    annot = tmp_path / "20260101_120000_tid7_zona_1_annot.jpg"
    assert annot.exists()


def test_save_evidence_sin_annot(tmp_path, sample_frame):
    frame = sample_frame(value=100)
    ctx = _ctx(tmp_path, frame, save_annotated=False)
    path = save_evidence(_job(frame), ctx)
    assert path.exists()
    assert not (tmp_path / "20260101_120000_tid7_zona_1_annot.jpg").exists()


def test_save_evidence_rtsp_dibuja_overlay_una_vez(tmp_path, sample_frame, monkeypatch):
    calls = {"n": 0}
    real = events.draw_overlay

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(events, "draw_overlay", counting)
    frame = sample_frame(value=100)
    ctx = _ctx(tmp_path, frame, save_annotated=True)
    save_evidence(_job(frame), ctx)
    assert calls["n"] == 1  # jpg y annot comparten el dibujo


def test_save_evidence_isapi_usa_snapshot(tmp_path, sample_frame, monkeypatch):
    jpeg = b"\xff\xd8isapi-bytes"
    monkeypatch.setattr(events, "fetch_isapi_snapshot", lambda *a, **k: jpeg)
    frame = sample_frame(value=100)
    ctx = _ctx(tmp_path, frame, evidence_mode="isapi", isapi_url="http://cam/pic", isapi_user="u", isapi_pass="p")
    path = save_evidence(_job(frame), ctx)
    assert path.read_bytes() == jpeg


def test_save_evidence_isapi_falla_y_hace_fallback(tmp_path, sample_frame, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("sin red")

    monkeypatch.setattr(events, "fetch_isapi_snapshot", boom)
    frame = sample_frame(value=100)
    ctx = _ctx(tmp_path, frame, evidence_mode="isapi", fallback_to_rtsp=True)
    path = save_evidence(_job(frame), ctx)
    assert path is not None
    assert path.read_bytes().startswith(b"\xff\xd8")


def test_save_evidence_isapi_falla_sin_fallback_retorna_none(tmp_path, sample_frame, monkeypatch):
    monkeypatch.setattr(events, "fetch_isapi_snapshot", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    frame = sample_frame(value=100)
    ctx = _ctx(tmp_path, frame, evidence_mode="isapi", fallback_to_rtsp=False)
    assert save_evidence(_job(frame), ctx) is None


def test_event_worker_envia_a_waha(tmp_path, sample_frame):
    frame = sample_frame(value=100)
    ctx = _ctx(tmp_path, frame)
    worker = EventWorker(ctx)
    worker.start()
    assert worker.submit(_job(frame))
    worker.stop()
    assert len(ctx.waha.calls) == 1
    chat_id, name, caption = ctx.waha.calls[0]
    assert chat_id == "grupo@g.us"
    assert name == "20260101_120000_tid7_zona_1.jpg"
    assert caption == "Alerta zona_1 3.2s 2026-01-01 12:00:00"


def test_event_worker_submit_cola_llena_dropea(tmp_path, sample_frame):
    frame = sample_frame(value=100)
    ctx = _ctx(tmp_path, frame)
    worker = EventWorker(ctx, maxsize=1)
    assert worker.submit(_job(frame))
    assert worker.submit(_job(frame)) is False  # no bloquea


def test_build_overlay_payload(tmp_path):
    payload = build_overlay_payload(
        "t", "zona_1", _roi(), (1080, 1920), (540, 960), [], {}
    )
    assert payload["roi"]["name"] == "zona_1"
    assert payload["frame"]["orig_hw"] == [1080, 1920]
    assert payload["frame"]["proc_hw"] == [540, 960]
    assert payload["tracks"] == []
    # serializable
    json.dumps(payload)


def test_build_overlay_payload_shape_none():
    payload = build_overlay_payload("t", "z", _roi(), None, None, [], {})
    assert payload["frame"]["orig_hw"] is None
    assert payload["frame"]["proc_hw"] is None
