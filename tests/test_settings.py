import numpy as np
import pytest

from roi_alert.settings import RunnerSettings

ENV_KEYS = [
    "CAM_RTSP_URL",
    "CAM_RTSP_URL_SUB",
    "EVENTS_DIR",
    "WAHA_URL",
    "WAHA_API_KEY",
    "WAHA_SESSION",
    "WAHA_CHAT_ID",
    "ISAPI_URL",
    "ISAPI_AUTH",
    "ISAPI_USER",
    "ISAPI_PASS",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def _cfg():
    return {
        "camera": {
            "process_width": 640,
            "fps_limit": 8,
            "reconnect_backoff_sec": 2,
            "capture_fps_limit": 20,
            "detect_every_n_frames": 2,
            "overlay_hz": 10,
            "rtsp_url": "rtsp://cfg/main",
        },
        "model": {"yolo_weights": "yolov8n.pt", "conf_thres": 0.35},
        "roi": {"name": "zona_1", "points": [[0, 0], [10, 0], [10, 10], [0, 10]]},
        "logic": {"dwell_seconds": 2.0, "cooldown_seconds": 60, "iou_match_thres": 0.3, "track_ttl_seconds": 2.0},
        "whatsapp": {"session": "default", "chat_id": "g@g.us", "caption_template": "A {roi} {dwell:.1f}s {ts}"},
        "storage": {"events_dir": "./events", "save_annotated": True},
        "evidence": {"mode": "rtsp", "timeout_s": 4, "fallback_to_rtsp": True},
        "isapi": {"url": "http://cam/pic", "auth": "digest"},
    }


def test_from_cfg_resuelve_campos():
    s = RunnerSettings.from_cfg(_cfg())
    assert s.rtsp == "rtsp://cfg/main"
    assert s.process_w == 640
    assert s.fps_limit == 8.0
    assert s.capture_fps_limit == 20.0
    assert s.detect_every == 2
    assert s.yolo_weights == "yolov8n.pt"
    assert s.conf_thres == 0.35
    assert s.roi_name == "zona_1"
    assert s.roi_points.dtype == np.int32
    assert s.dwell_sec == 2.0
    assert s.cooldown_sec == 60.0
    assert s.chat_id == "g@g.us"
    assert s.evidence_mode == "rtsp"
    assert s.warnings == []


def test_capture_fps_default_es_max_12(monkeypatch):
    cfg = _cfg()
    del cfg["camera"]["capture_fps_limit"]
    cfg["camera"]["fps_limit"] = 5
    s = RunnerSettings.from_cfg(cfg)
    assert s.capture_fps_limit == 12.0
    cfg["camera"]["fps_limit"] = 20
    s2 = RunnerSettings.from_cfg(cfg)
    assert s2.capture_fps_limit == 20.0


def test_env_gana_a_config_en_rtsp(monkeypatch):
    monkeypatch.setenv("CAM_RTSP_URL", "rtsp://env/main")
    monkeypatch.setenv("CAM_RTSP_URL_SUB", "rtsp://env/sub")
    s = RunnerSettings.from_cfg(_cfg())
    assert s.rtsp == "rtsp://env/sub"


def test_env_gana_en_waha(monkeypatch):
    monkeypatch.setenv("WAHA_URL", "http://env:9999")
    monkeypatch.setenv("WAHA_API_KEY", "secret")
    monkeypatch.setenv("WAHA_SESSION", "sess2")
    monkeypatch.setenv("WAHA_CHAT_ID", "env@g.us")
    s = RunnerSettings.from_cfg(_cfg())
    assert s.waha_base == "http://env:9999"
    assert s.waha_key == "secret"
    assert s.waha_session == "sess2"
    assert s.chat_id == "env@g.us"


def test_falta_chat_id_lanza_systemexit():
    cfg = _cfg()
    cfg["whatsapp"]["chat_id"] = ""
    with pytest.raises(SystemExit, match="chat_id"):
        RunnerSettings.from_cfg(cfg)


def test_isapi_env_y_fallback(monkeypatch):
    cfg = _cfg()
    cfg["evidence"]["mode"] = "isapi"
    # sin credenciales -> fallback a rtsp con warning
    s = RunnerSettings.from_cfg(cfg)
    assert s.evidence_mode == "rtsp"
    assert any("isapi" in w for w in s.warnings)

    # con credenciales por env se mantiene isapi
    monkeypatch.setenv("ISAPI_USER", "admin")
    monkeypatch.setenv("ISAPI_PASS", "pw")
    monkeypatch.setenv("ISAPI_URL", "http://env/pic")
    s2 = RunnerSettings.from_cfg(cfg)
    assert s2.evidence_mode == "isapi"
    assert s2.isapi_user == "admin"
    assert s2.isapi_pass == "pw"
    assert s2.isapi_url == "http://env/pic"


def test_events_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("EVENTS_DIR", str(tmp_path / "ev"))
    s = RunnerSettings.from_cfg(_cfg())
    assert s.events_dir == tmp_path / "ev"


def test_roi_from_json_flags():
    cfg = _cfg()
    cfg["roi"]["from_json"] = True
    cfg["roi"]["json_path"] = "/tmp/roi.json"
    s = RunnerSettings.from_cfg(cfg)
    assert s.roi_from_json is True
    assert s.roi_json_path == "/tmp/roi.json"
    assert s.roi_points.tolist() == [[0, 0], [10, 0], [10, 10], [0, 10]]
