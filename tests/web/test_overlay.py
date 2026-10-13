import json

import numpy as np

from webapp.services.overlay import (
    build_status_lines,
    draw_roi,
    extract_roi_poly,
    extract_roi_rect,
    read_json,
    render_overlay,
    resolve_base_width,
)


def test_read_json_valido(tmp_path):
    p = tmp_path / "o.json"
    p.write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert read_json(p) == {"a": 1}


def test_read_json_inexistente_devuelve_vacio(tmp_path):
    assert read_json(tmp_path / "nope.json") == {}


def test_read_json_malformado_devuelve_vacio(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{no-json", encoding="utf-8")
    assert read_json(p) == {}


def test_extract_roi_poly_desde_roi_points():
    o = {"roi": {"points": [[0, 0], [10, 0], [10, 10]]}}
    arr = extract_roi_poly(o)
    assert arr.shape == (3, 2)
    assert arr.dtype == np.int32


def test_extract_roi_poly_variantes():
    assert extract_roi_poly({"roi_points": [[0, 0], [1, 1], [2, 2]]}).shape == (3, 2)
    assert extract_roi_poly({"points": [[0, 0], [1, 1], [2, 2]]}).shape == (3, 2)


def test_extract_roi_poly_insuficiente_o_invalido():
    assert extract_roi_poly({"roi": {"points": [[0, 0], [1, 1]]}}) is None
    assert extract_roi_poly({"roi": {"points": "no"}}) is None
    assert extract_roi_poly({}) is None
    assert extract_roi_poly({"roi": {"points": [["a"], [1, 2], [3, 4]]}}) is None


def test_extract_roi_rect_xyxy():
    assert extract_roi_rect({"roi_xyxy": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}}) == (1, 2, 3, 4)


def test_extract_roi_rect_xywh():
    assert extract_roi_rect({"roi_xywh": {"x": 1, "y": 2, "w": 10, "h": 20}}) == (1, 2, 11, 22)


def test_extract_roi_rect_ausente():
    assert extract_roi_rect({}) is None
    assert extract_roi_rect({"roi_xyxy": {"x1": 1}}) is None


def test_resolve_base_width_prioridad_status():
    assert resolve_base_width({"process_width": 1280}, {"width": 640}, 320) == 1280
    assert resolve_base_width({}, {"process_w": 960}, 320) == 960
    assert resolve_base_width({}, {}, 320) == 320
    assert resolve_base_width({"width": "basura"}, {}, 320) == 320


def test_build_status_lines_completo():
    lines = build_status_lines({"ts": "t", "inside_count": 2, "max_dwell_s": 3.0, "infer_fps": 5.0}, 42)
    assert "ts: t" in lines
    assert "inside: 2" in lines
    assert "max_dwell_s: 3.0" in lines
    assert "infer_fps: 5.0" in lines
    assert "grab_age_ms: 42" in lines


def test_build_status_lines_minimo():
    lines = build_status_lines({}, 7)
    assert lines == ["grab_age_ms: 7"]


def test_draw_roi_poligono_cambia_frame(sample_frame):
    frame = sample_frame(value=0)
    poly = np.array([[0, 0], [10, 0], [10, 10]], dtype=np.int32)
    draw_roi(frame, poly, None, scale=1.0)
    assert frame.any()


def test_draw_roi_rect_cuando_no_hay_poly(sample_frame):
    frame = sample_frame(value=0)
    draw_roi(frame, None, (0, 0, 10, 10), scale=1.0)
    assert frame.any()


def test_draw_roi_poly_tiene_prioridad(sample_frame):
    frame = sample_frame(value=0)
    poly = np.array([[0, 0], [5, 0], [5, 5]], dtype=np.int32)
    draw_roi(frame, poly, (20, 20, 30, 30), scale=1.0)
    # el rect se ignora si hay poly
    assert frame.any()


def test_render_overlay_dibuja_y_muta(sample_frame):
    frame = sample_frame(value=0)
    overlay = {"roi": {"points": [[0, 0], [50, 0], [50, 50]]}}
    out = render_overlay(
        frame, overlay, {}, preview_width=96, stale_ms_warn=1500, meta_ts_monotonic=0.0, now_fn=lambda: 0.0
    )
    assert out is frame
    assert frame.any()


def test_render_overlay_borde_stale(sample_frame):
    frame = sample_frame(value=0)
    out = render_overlay(
        frame, {}, {}, preview_width=96, stale_ms_warn=100, meta_ts_monotonic=1.0, now_fn=lambda: 5.0
    )
    # age = 4000ms > 100 -> dibuja borde rojo (BGR 0,0,255) en (2,2)
    assert tuple(out[2, 2]) == (0, 0, 255)


def test_render_overlay_sin_stale_sin_borde(sample_frame):
    frame = sample_frame(value=0)
    out = render_overlay(
        frame, {}, {}, preview_width=96, stale_ms_warn=100000, meta_ts_monotonic=1.0, now_fn=lambda: 1.0
    )
    assert tuple(out[2, 2]) == (0, 0, 0)
