import json
from pathlib import Path
from typing import Tuple

import numpy as np


def load_roi_from_json(json_path: Path, target_proc_shape_hw: Tuple[int, int]) -> np.ndarray:
    """
    Soporta roi.json tipo:
    {
      "roi_xyxy": {"x1":..., "y1":..., "x2":..., "y2":...},
      "image_shape_hw": [1080,1920]
    }
    Escala al tamaño target (procesado).
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    img_h, img_w = data.get("image_shape_hw", [0, 0])
    if not img_h or not img_w:
        raise ValueError("roi.json no trae image_shape_hw válido")

    roi_xyxy = data.get("roi_xyxy")
    if not roi_xyxy:
        raise ValueError("roi.json no trae roi_xyxy")

    x1 = float(roi_xyxy["x1"]); y1 = float(roi_xyxy["y1"])
    x2 = float(roi_xyxy["x2"]); y2 = float(roi_xyxy["y2"])

    tgt_h, tgt_w = target_proc_shape_hw
    sx = tgt_w / float(img_w)
    sy = tgt_h / float(img_h)

    pts = np.array([
        [x1 * sx, y1 * sy],
        [x2 * sx, y1 * sy],
        [x2 * sx, y2 * sy],
        [x1 * sx, y2 * sy],
    ], dtype=np.int32)

    return pts