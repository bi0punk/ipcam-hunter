"""Reglas de dwell/ROI: pura, sin dependencias de cv2/YOLO en el camino crítico.

Centraliza la lógica que antes vivía dentro del bucle de `runner.run`:
  - cuánto tiempo lleva cada track dentro del ROI (dwell)
  - cuándo disparar una alerta (dwell >= dwell_sec + cooldown anti-spam)
  - reset al salir del ROI
"""
from dataclasses import dataclass

import numpy as np

from .tracker import Track
from .vision import centroid, point_in_roi


@dataclass
class DwellResult:
    dwell_map: dict[int, float]
    triggered_track: Track | None = None
    triggered_dwell: float = 0.0
    new_alert_ts: float = 0.0


def evaluate_dwell(
    tracks: list[Track],
    roi_points: np.ndarray,
    ts: float,
    dwell_sec: float,
    cooldown_sec: float,
    last_alert_ts: float,
) -> DwellResult:
    """Evalúa la regla de alerta por estancia en ROI.

    Muta `in_roi_since`/`fired` de cada track (mismo comportamiento que el
    bucle original). Devuelve cuál track disparó (si hubo) y el nuevo
    `last_alert_ts` a persistir.
    """
    dwell_map: dict[int, float] = {}
    triggered_track: Track | None = None
    triggered_dwell = 0.0
    new_alert_ts = last_alert_ts

    for tr in tracks:
        inside = point_in_roi(centroid(tr.bbox), roi_points)

        if inside:
            if tr.in_roi_since is None:
                tr.in_roi_since = ts
                tr.fired = False
            dwell = ts - tr.in_roi_since
            dwell_map[tr.tid] = dwell

            if (not tr.fired) and (dwell >= dwell_sec) and (ts - new_alert_ts) >= cooldown_sec:
                triggered_track = tr
                triggered_dwell = dwell
                tr.fired = True
                new_alert_ts = ts
        else:
            tr.in_roi_since = None
            tr.fired = False

    return DwellResult(
        dwell_map=dwell_map,
        triggered_track=triggered_track,
        triggered_dwell=triggered_dwell,
        new_alert_ts=new_alert_ts,
    )


def track_to_overlay(tr: Track, roi_points: np.ndarray, dwell_map: dict[int, float]) -> dict:
    """Serializa un track para overlay.json."""
    c = centroid(tr.bbox)
    return {
        "id": tr.tid,
        "bbox": [round(tr.bbox[0], 2), round(tr.bbox[1], 2), round(tr.bbox[2], 2), round(tr.bbox[3], 2)],
        "centroid": [int(c[0]), int(c[1])],
        "inside": bool(point_in_roi(c, roi_points)),
        "dwell_s": round(dwell_map.get(tr.tid, 0.0), 2),
    }


def inside_stats(tracks: list[Track], dwell_map: dict[int, float]) -> tuple[int, float]:
    """Cuenta tracks dentro del ROI y el máximo dwell (para status.json)."""
    inside_count = 0
    max_dwell = 0.0
    for tr in tracks:
        d = dwell_map.get(tr.tid, 0.0)
        if d > 0:
            inside_count += 1
            max_dwell = max(max_dwell, d)
    return inside_count, max_dwell
