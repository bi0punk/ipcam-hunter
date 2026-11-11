from pathlib import Path

from roi_alert.cfg import load_cfg, pick_rtsp, resolve_events_dir


def test_load_cfg_parsea_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("camera:\n  process_width: 640\n", encoding="utf-8")
    assert load_cfg(p) == {"camera": {"process_width": 640}}


def test_load_cfg_vacio_devuelve_dict(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("", encoding="utf-8")
    assert load_cfg(p) == {}


def test_resolve_events_dir_env_gana(monkeypatch, tmp_path):
    monkeypatch.setenv("EVENTS_DIR", str(tmp_path / "mi_events"))
    assert resolve_events_dir("ignored") == tmp_path / "mi_events"


def test_resolve_events_dir_docker_cuando_existe(monkeypatch):
    monkeypatch.delenv("EVENTS_DIR", raising=False)
    existe = lambda path: path == Path("/data/events")  # noqa: E731
    assert resolve_events_dir("./events", exists=existe) == Path("/data/events")
    assert resolve_events_dir("events", exists=existe) == Path("/data/events")


def test_resolve_events_dir_absoluta_no_se_reescribe(monkeypatch):
    monkeypatch.delenv("EVENTS_DIR", raising=False)
    p = Path("/var/custom/events")
    assert resolve_events_dir(str(p), exists=lambda _path: True) == p


def test_pick_rtsp_env_principal(monkeypatch):
    monkeypatch.delenv("CAM_RTSP_URL", raising=False)
    monkeypatch.delenv("CAM_RTSP_URL_SUB", raising=False)
    monkeypatch.setenv("CAM_RTSP_URL", "rtsp://user@cam/main")
    cfg = {}
    assert pick_rtsp(cfg) == "rtsp://user@cam/main"


def test_pick_rtsp_sub_gana_a_principal(monkeypatch):
    monkeypatch.setenv("CAM_RTSP_URL", "rtsp:///main")
    monkeypatch.setenv("CAM_RTSP_URL_SUB", "rtsp:///sub")
    assert pick_rtsp({}) == "rtsp:///sub"


def test_pick_rtsp_fallback_a_config():
    cfg = {"camera": {"rtsp_url": "rtsp:///from-config", "rtsp_url_sub": "rtsp:///from-config-sub"}}
    assert pick_rtsp(cfg) == "rtsp:///from-config-sub"


def test_pick_rtsp_sin_nada(monkeypatch):
    monkeypatch.delenv("CAM_RTSP_URL", raising=False)
    monkeypatch.delenv("CAM_RTSP_URL_SUB", raising=False)
    assert pick_rtsp({}) == ""
    assert pick_rtsp({"camera": {}}) == ""


def test_pick_rtsp_rtsp_literal_de_config(monkeypatch):
    monkeypatch.delenv("CAM_RTSP_URL", raising=False)
    monkeypatch.delenv("CAM_RTSP_URL_SUB", raising=False)
    cfg = {"camera": {"rtsp_url": "rtsp://cam/stream"}}
    assert pick_rtsp(cfg) == "rtsp://cam/stream"
