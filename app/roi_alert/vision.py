from typing import Dict, List, Tuple

import cv2
import numpy as np

from .types import BBox


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def centroid(bb: BBox) -> Tuple[int, int]:
    x1, y1, x2, y2 = bb
    return (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))


def point_in_roi(pt: Tuple[int, int], roi_pts: np.ndarray) -> bool:
    return cv2.pointPolygonTest(roi_pts, pt, False) >= 0


def draw_overlay(img: np.ndarray, roi_pts: np.ndarray, tracks: List, dwell_map: Dict[int, float], iso_time_fn) -> np.ndarray:
    """
    tracks: lista de Track (evitamos import circular aquí)
    iso_time_fn: función para timestamp en overlay
    """
    out = img.copy()
    cv2.polylines(out, [roi_pts.astype(np.int32)], True, (0, 255, 0), 2)
    for tr in tracks:
        x1, y1, x2, y2 = map(int, tr.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)
        d = dwell_map.get(tr.tid, 0.0)
        cv2.putText(out, f"id={tr.tid} dwell={d:.1f}s", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(out, iso_time_fn(), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out