"""Guardas de seguridad: el repo no debe contener credenciales/endpoints reales."""
import re
from pathlib import Path

from roi_alert.settings import RunnerSettings

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "app" / "config.yaml"
ENV_EXAMPLE = ROOT / ".example.env"

FORBIDDEN = ["192.168.", "password=", "passwd", "Bearer "]
# ID real de grupo WhatsApp: muchos digitos seguidos de @g.us
WA_GROUP_ID = re.compile(r"\d{5,}@g\.us")


def test_config_yaml_sin_credenciales():
    text = CONFIG.read_text(encoding="utf-8")
    assert 'chat_id: ""' in text
    assert 'url: ""' in text
    for bad in FORBIDDEN:
        assert bad not in text, f"config.yaml contiene {bad!r}"
    assert not WA_GROUP_ID.search(text)


def test_example_env_sin_credenciales_reales():
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    for bad in FORBIDDEN:
        assert bad not in text, f".example.env contiene {bad!r}"
    assert not WA_GROUP_ID.search(text)
    # valores sensibles deben venir vacios en la plantilla
    for key in ("CAM_RTSP_URL=", "ISAPI_URL=", "ISAPI_USER=", "ISAPI_PASS=", "WAHA_API_KEY=", "WAHA_CHAT_ID="):
        assert key in text, f"falta {key} en .example.env"


def test_default_seguro_evidencia_rtsp(monkeypatch):
    for k in ("ISAPI_URL", "ISAPI_USER", "ISAPI_PASS", "EVENTS_DIR", "WAHA_CHAT_ID"):
        monkeypatch.delenv(k, raising=False)
    cfg = {
        "camera": {"process_width": 640, "fps_limit": 8, "reconnect_backoff_sec": 2},
        "model": {"yolo_weights": "yolov8n.pt", "conf_thres": 0.35},
        "roi": {"name": "z", "points": [[0, 0], [1, 0], [1, 1]]},
        "logic": {"dwell_seconds": 2, "cooldown_seconds": 60, "iou_match_thres": 0.3, "track_ttl_seconds": 2},
        "whatsapp": {"chat_id": "x@g.us", "caption_template": "c"},
        "storage": {"events_dir": "./events"},
        "evidence": {"mode": "isapi"},
        "isapi": {},
    }
    s = RunnerSettings.from_cfg(cfg)
    # sin credenciales ISAPI no se habilita isapi: default seguro rtsp
    assert s.evidence_mode == "rtsp"
    assert any("isapi" in w for w in s.warnings)


def test_env_ignorado_en_git():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore
    assert "waha-sessions" in gitignore
