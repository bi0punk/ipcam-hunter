"""Configuración unificada del worker (env + config.yaml).

Único punto de resolución de settings para `roi_alert`: RTSP, cámara, modelo,
ROI, lógica de dwell, storage, WhatsApp (WAHA) y evidencia ISAPI. Prioriza
variables de entorno sobre `config.yaml` para todo lo sensible.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .cfg import load_cfg, pick_rtsp, resolve_events_dir


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, "").strip() or default


@dataclass
class RunnerSettings:
    rtsp: str
    process_w: int
    fps_limit: float
    capture_fps_limit: float
    backoff: float
    detect_every: int
    overlay_hz: float
    yolo_weights: str
    conf_thres: float
    roi_name: str
    roi_points: np.ndarray
    roi_from_json: bool
    roi_json_path: str
    dwell_sec: float
    cooldown_sec: float
    match_thres: float
    ttl_sec: float
    events_dir: Path
    save_annotated: bool
    waha_base: str
    waha_key: str
    waha_session: str
    chat_id: str
    caption_template: str
    evidence_mode: str
    evidence_timeout_s: float
    fallback_to_rtsp: bool
    isapi_url: str
    isapi_auth: str
    isapi_user: str
    isapi_pass: str
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_cfg(cls, cfg: dict, cfg_path: Path | None = None) -> "RunnerSettings":
        camera = cfg["camera"]
        whatsapp = cfg.get("whatsapp", {}) or {}
        evidence_cfg = cfg.get("evidence", {}) or {}
        isapi_cfg = cfg.get("isapi", {}) or {}

        fps_limit = float(camera["fps_limit"])
        warnings: list[str] = []

        evidence_mode = str(evidence_cfg.get("mode", "rtsp")).strip().lower()
        isapi_url = _env("ISAPI_URL") or str(isapi_cfg.get("url", "")).strip()
        isapi_auth = _env("ISAPI_AUTH").lower() or str(isapi_cfg.get("auth", "digest")).strip().lower()
        isapi_user = _env("ISAPI_USER")
        isapi_pass = _env("ISAPI_PASS")

        if evidence_mode == "isapi" and (not isapi_url or not isapi_user or not isapi_pass):
            warnings.append("[warn] evidence.mode=isapi pero faltan ISAPI_URL/USER/PASS -> usando rtsp fallback")
            evidence_mode = "rtsp"

        chat_id = _env("WAHA_CHAT_ID") or str(whatsapp.get("chat_id", "")).strip()
        if not chat_id:
            raise SystemExit("Falta chat_id. Define WAHA_CHAT_ID o whatsapp.chat_id en config.yaml")

        return cls(
            rtsp=pick_rtsp(cfg),
            process_w=int(camera["process_width"]),
            fps_limit=fps_limit,
            capture_fps_limit=float(camera.get("capture_fps_limit", max(12.0, fps_limit))),
            backoff=float(camera["reconnect_backoff_sec"]),
            detect_every=int(camera.get("detect_every_n_frames", 1)),
            overlay_hz=float(camera.get("overlay_hz", 10)),
            yolo_weights=cfg["model"]["yolo_weights"],
            conf_thres=float(cfg["model"]["conf_thres"]),
            roi_name=cfg["roi"].get("name", "roi"),
            roi_points=np.array(cfg["roi"]["points"], dtype=np.int32),
            roi_from_json=bool((cfg.get("roi", {}) or {}).get("from_json", False)),
            roi_json_path=str((cfg.get("roi", {}) or {}).get("json_path", "roi.json")),
            dwell_sec=float(cfg["logic"]["dwell_seconds"]),
            cooldown_sec=float(cfg["logic"]["cooldown_seconds"]),
            match_thres=float(cfg["logic"]["iou_match_thres"]),
            ttl_sec=float(cfg["logic"]["track_ttl_seconds"]),
            events_dir=resolve_events_dir(str(cfg["storage"]["events_dir"])),
            save_annotated=bool(cfg["storage"].get("save_annotated", True)),
            waha_base=_env("WAHA_URL") or str(whatsapp.get("waha_url", "http://waha:3000")),
            waha_key=_env("WAHA_API_KEY") or str(whatsapp.get("api_key", "")),
            waha_session=_env("WAHA_SESSION") or str(whatsapp.get("session", "default")),
            chat_id=chat_id,
            caption_template=cfg["whatsapp"]["caption_template"],
            evidence_mode=evidence_mode,
            evidence_timeout_s=float(evidence_cfg.get("timeout_s", 4.0)),
            fallback_to_rtsp=bool(evidence_cfg.get("fallback_to_rtsp", True)),
            isapi_url=isapi_url,
            isapi_auth=isapi_auth,
            isapi_user=isapi_user,
            isapi_pass=isapi_pass,
            warnings=warnings,
        )

    @classmethod
    def from_file(cls, cfg_path: Path) -> "RunnerSettings":
        return cls.from_cfg(load_cfg(cfg_path), cfg_path=cfg_path)
