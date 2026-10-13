"""Overlay y extracción de ROI para el dashboard MJPEG.

Funciones puras (sin estado del hub) para poder testear la generación de
frames sin levantar captura RTSP.
"""
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def extract_roi_poly(overlay: dict[str, Any]) -> np.ndarray | None:
    """Soporta overlay["roi"]["points"], overlay["roi_points"], overlay["points"]."""
    pts = None
    if isinstance(overlay.get("roi"), dict):
        pts = overlay["roi"].get("points")
    if pts is None:
        pts = overlay.get("roi_points")
    if pts is None:
        pts = overlay.get("points")

    if not pts or not isinstance(pts, list) or len(pts) < 3:
        return None
    try:
        return np.array([[int(p[0]), int(p[1])] for p in pts], dtype=np.int32)
    except Exception:
        return None


def extract_roi_rect(overlay: dict[str, Any]) -> tuple[int, int, int, int] | None:
    """Soporta roi_xyxy {x1,y1,x2,y2} y roi_xywh {x,y,w,h}."""
    r = overlay.get("roi_xyxy")
    if isinstance(r, dict) and all(k in r for k in ("x1", "y1", "x2", "y2")):
        return int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
    r = overlay.get("roi_xywh")
    if isinstance(r, dict) and all(k in r for k in ("x", "y", "w", "h")):
        x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
        return x, y, x + w, y + h
    return None


def resolve_base_width(status: dict[str, Any], overlay: dict[str, Any], default: int) -> int:
    """Ancho de referencia (process_width) para escalar el ROI al preview."""
    base_w = None
    for k in ("process_width", "process_w", "width"):
        if k in status:
            base_w = status.get(k)
            break
        if k in overlay:
            base_w = overlay.get(k)
            break
    try:
        return int(base_w) if base_w else int(default)
    except Exception:
        return int(default)


def draw_roi(out: np.ndarray, poly: np.ndarray | None, rect: tuple[int, int, int, int] | None, scale: float) -> None:
    if poly is not None:
        pts = (poly.astype(np.float32) * scale).astype(np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 0), 2)
    if rect is not None and poly is None:
        x1, y1, x2, y2 = (int(v * scale) for v in rect)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)


def build_status_lines(status: dict[str, Any], age_ms: int) -> list[str]:
    lines = []
    ts = status.get("ts") or ""
    if ts:
        lines.append(f"ts: {ts}")
    inside = status.get("inside_count")
    if inside is not None:
        lines.append(f"inside: {inside}")
    md = status.get("max_dwell_s")
    if md is not None:
        lines.append(f"max_dwell_s: {md}")
    fps = status.get("infer_fps")
    if fps is not None:
        lines.append(f"infer_fps: {fps}")
    lines.append(f"grab_age_ms: {age_ms}")
    return lines


def render_overlay(
    frame: np.ndarray,
    overlay: dict[str, Any],
    status: dict[str, Any],
    *,
    preview_width: int,
    stale_ms_warn: int,
    meta_ts_monotonic: float,
    now_fn=time.monotonic,
) -> np.ndarray:
    """Dibuja ROI + stats + estado de stream sobre una copia del frame."""
    out = frame
    poly = extract_roi_poly(overlay)
    rect = extract_roi_rect(overlay)

    base_w = resolve_base_width(status, overlay, preview_width)
    scale = float(preview_width) / float(base_w) if base_w > 0 else 1.0
    draw_roi(out, poly, rect, scale)

    age_ms = int((now_fn() - meta_ts_monotonic) * 1000.0) if meta_ts_monotonic else 999999
    if age_ms > stale_ms_warn:
        h, w = out.shape[:2]
        cv2.rectangle(out, (2, 2), (w - 3, h - 3), (0, 0, 255), 6)

    y = 22
    for s in build_status_lines(status, age_ms):
        cv2.putText(out, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20
    return out
