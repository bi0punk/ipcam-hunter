import numpy as np
import pytest

from roi_alert.vision import centroid, draw_overlay, iou, point_in_roi


# ---------- iou ----------

def test_iou_box_identica_es_1():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_cajas_disjuntas_es_0():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_solapamiento_parcial():
    a = (0, 0, 10, 10)
    b = (5, 0, 15, 10)  # mitad derecha de a -> inter 5x10, union 15x10
    assert iou(a, b) == pytest.approx(50.0 / 150.0)


def test_iou_contencion_total():
    a = (0, 0, 10, 10)
    b = (2, 2, 8, 8)
    # inter = area de b = 36, union = area de a = 100
    assert iou(a, b) == pytest.approx(36.0 / 100.0)


def test_iou_bbox_de_area_cero():
    assert iou((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0
    assert iou((0, 0, 10, 10), (5, 5, 5, 5)) == 0.0
    assert iou((5, 5, 5, 5), (5, 5, 5, 5)) == 0.0


def test_iou_solo_tocan_bordes():
    # comparten arista x=10 -> area de interseccion 0
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


# ---------- centroid ----------

def test_centroid_entero():
    assert centroid((0, 0, 10, 10)) == (5, 5)


def test_centroid_redondea_al_entero_mas_cercano():
    assert centroid((1, 1, 2, 2)) == (1, 1)  # (1.5, 1.5) -> int() trunca a 1
    assert centroid((0, 0, 1, 1)) == (0, 0)


def test_centroid_bbox_no_alineada_a_pixel():
    assert centroid((1.2, 3.8, 9.2, 11.1)) == (5, 7)


# ---------- point_in_roi ----------

def _roi_square():
    return np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int32)


def test_punto_dentro_roi():
    assert point_in_roi((5, 5), _roi_square())


def test_punto_fuera_roi():
    assert not point_in_roi((50, 50), _roi_square())
    assert not point_in_roi((-5, -5), _roi_square())


def test_punto_en_borde_cuenta_como_dentro():
    # pointPolygonTest >= 0 incluye el contorno
    assert point_in_roi((0, 5), _roi_square())
    assert point_in_roi((10, 10), _roi_square())


def test_punto_en_esquina():
    assert point_in_roi((0, 0), _roi_square())


# ---------- draw_overlay ----------

def test_draw_overlay_devuelve_copia_sin_mutar_input(sample_frame):
    frame = sample_frame(value=127)
    roi = _roi_square()
    out = draw_overlay(frame, roi, [], {}, lambda: "2026-01-01 00:00:00")
    assert out is not frame
    # sin tracks: el texto de timestamp y el poligono cambian el contenido
    assert not np.array_equal(out, frame)


def test_draw_overlay_con_tracks(sample_frame):
    frame = sample_frame(value=100)
    tracks = [type("T", (), {"tid": 3, "bbox": (2, 2, 30, 60), "fired": True})()]
    roi = _roi_square()
    out = draw_overlay(frame, roi, tracks, {3: 4.5}, lambda: "t")
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)


def test_draw_overlay_con_dwell_usar_dict(sample_frame):
    frame = sample_frame(value=80)
    roi = _roi_square()
    tr = type("T", (), {"tid": 9, "bbox": (0, 0, 20, 20), "fired": False})()
    out = draw_overlay(frame, roi, [tr], {9: 12.34}, lambda: "x")
    assert out.shape == frame.shape