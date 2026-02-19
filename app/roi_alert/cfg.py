import os
from pathlib import Path

import yaml


def load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_events_dir(cfg_events_dir: str) -> Path:
    env_events = os.getenv("EVENTS_DIR", "").strip()
    if env_events:
        return Path(env_events)

    p = Path(cfg_events_dir)
    docker_events = Path("/data/events")
    if not p.is_absolute() and docker_events.exists():
        if cfg_events_dir.strip() in ("./events", "events"):
            return docker_events
    return p


def pick_rtsp(cfg: dict) -> str:
    cam = (cfg.get("camera", {}) or {})
    rtsp_base = os.getenv("CAM_RTSP_URL", "").strip() or cam.get("rtsp_url", "")
    rtsp_sub = os.getenv("CAM_RTSP_URL_SUB", "").strip() or cam.get("rtsp_url_sub", "")
    rtsp = rtsp_sub or rtsp_base
    return (rtsp or "").strip()