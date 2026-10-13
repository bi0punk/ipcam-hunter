import json
import math

import numpy as np
import pytest

from roi_alert.roi_json import load_roi_from_json


def _write_roi(tmp_path, roi_xyxy=None, image_shape_hw=None):
    data = {}
    if image_shape_hw is not None:
        data["image_shape_hw"] = list(image_shape_hw)
    if roi_xyxy is not None:
        data["roi_xyxy"] = roi_xyxy
    p = tmp_path / "roi.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_escala_de_1920x1080_a_proc(tmp_path):
    roi = {"x1": 0, "y1": 0, "x2": 1920, "y2": 1080}
    path = _write_roi(tmp_path, roi_xyxy=roi, image_shape_hw=(1080, 1920))
    pts = load_roi_from_json(path, target_proc_shape_hw=(540, 960))
    expected = np.array(
        [[0, 0], [960, 0], [960, 540], [0, 540]],
        dtype=np.int32,
    )
    assert pts.dtype == np.int32
    assert pts.shape == (4, 2)
    np.testing.assert_array_equal(pts, expected)


def test_escala_con_factor_no_trivial(tmp_path):
    # ROI centrado 800x600 en 2000x1500, target 1000x750
    roi = {"x1": 600, "y1": 450, "x2": 1400, "y2": 1050}
    path = _write_roi(tmp_path, roi_xyxy=roi, image_shape_hw=(1500, 2000))
    pts = load_roi_from_json(path, target_proc_shape_hw=(750, 1000))
    expected = np.array(
        [[300, 225], [700, 225], [700, 525], [300, 525]],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(pts, expected)


def test_sin_image_shape_hw_lanza_valueerror(tmp_path):
    path = _write_roi(tmp_path, roi_xyxy={"x1": 0, "y1": 0, "x2": 640, "y2": 480})
    with pytest.raises(ValueError, match="image_shape_hw"):
        load_roi_from_json(path, (320, 320))


def test_image_shape_hw_cero_lanza_valueerror(tmp_path):
    path = _write_roi(tmp_path, roi_xyxy={"x1": 0, "y1": 0, "x2": 640, "y2": 480}, image_shape_hw=(0, 0))
    with pytest.raises(ValueError, match="image_shape_hw"):
        load_roi_from_json(path, (320, 320))


def test_sin_roi_xyxy_lanza_valueerror(tmp_path):
    path = _write_roi(tmp_path, image_shape_hw=(1080, 1920))
    with pytest.raises(ValueError, match="roi_xyxy"):
        load_roi_from_json(path, (320, 320))


def test_roi_xyxy_vacio_lanza_valueerror(tmp_path):
    # {} es falsy -> mismo error de validación
    path = _write_roi(tmp_path, roi_xyxy={}, image_shape_hw=(1080, 1920))
    with pytest.raises(ValueError, match="roi_xyxy"):
        load_roi_from_json(path, (320, 320))


def test_claves_incompletas_en_roi_xyxy_lanzan_keyerror(tmp_path):
    path = _write_roi(tmp_path, roi_xyxy={"x1": 10, "y1": 10, "x2": 20}, image_shape_hw=(1080, 1920))
    with pytest.raises(KeyError):
        load_roi_from_json(path, (320, 320))


def test_coords_negativas_con_escala(tmp_path):
    roi = {"x1": -100, "y1": -50, "x2": 100, "y2": 50}
    path = _write_roi(tmp_path, roi_xyxy=roi, image_shape_hw=(1000, 1000))
    pts = load_roi_from_json(path, target_proc_shape_hw=(500, 500))
    np.testing.assert_array_equal(pts, np.array([[-50, -25], [50, -25], [50, 25], [-50, 25]], dtype=np.int32))


def test_floats_como_coordenadas(tmp_path):
    roi = {"x1": 1.5, "y1": 2.5, "x2": 3.5, "y2": 4.5}
    path = _write_roi(tmp_path, roi_xyxy=roi, image_shape_hw=(10, 10))
    pts = load_roi_from_json(path, target_proc_shape_hw=(100, 100))
    assert not np.any(np.isnan(pts))
    assert math.isclose(float(pts[0, 0]), 15.0)
    assert math.isclose(float(pts[1, 0]), 35.0)
