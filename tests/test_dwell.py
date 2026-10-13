import numpy as np

from roi_alert.dwell import DwellResult, evaluate_dwell, inside_stats, track_to_overlay
from roi_alert.tracker import Track


def _roi():
    return np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32)


def _track(tid, bbox, in_roi_since=None, fired=False):
    return Track(tid=tid, bbox=bbox, last_seen=0.0, in_roi_since=in_roi_since, fired=fired)


def test_sin_tracks_no_dispara():
    r = evaluate_dwell([], _roi(), ts=10.0, dwell_sec=2.0, cooldown_sec=60.0, last_alert_ts=0.0)
    assert isinstance(r, DwellResult)
    assert r.dwell_map == {}
    assert r.triggered_track is None
    assert r.new_alert_ts == 0.0


def test_entrada_marca_in_roi_since_y_acumula_dwell():
    tr = _track(1, (40, 40, 60, 60))  # centro (50,50) dentro
    r = evaluate_dwell([tr], _roi(), ts=100.0, dwell_sec=5.0, cooldown_sec=60.0, last_alert_ts=0.0)
    assert tr.in_roi_since == 100.0
    assert r.dwell_map[1] == 0.0
    assert r.triggered_track is None


def test_dwell_no_alcanza_umbral_no_dispara():
    tr = _track(1, (40, 40, 60, 60), in_roi_since=100.0)
    r = evaluate_dwell([tr], _roi(), ts=101.5, dwell_sec=2.0, cooldown_sec=60.0, last_alert_ts=0.0)
    assert r.dwell_map[1] == 1.5
    assert r.triggered_track is None
    assert not tr.fired


def test_dwell_supera_umbral_dispara_y_marca_fired():
    tr = _track(1, (40, 40, 60, 60), in_roi_since=100.0)
    r = evaluate_dwell([tr], _roi(), ts=102.0, dwell_sec=2.0, cooldown_sec=60.0, last_alert_ts=0.0)
    assert r.triggered_track is tr
    assert r.triggered_dwell == 2.0
    assert tr.fired
    assert r.new_alert_ts == 102.0


def test_fired_no_re_dispara_mientras_sigue_dentro():
    tr = _track(1, (40, 40, 60, 60), in_roi_since=100.0, fired=True)
    r = evaluate_dwell([tr], _roi(), ts=110.0, dwell_sec=2.0, cooldown_sec=60.0, last_alert_ts=102.0)
    assert r.triggered_track is None
    assert r.dwell_map[1] == 10.0


def test_salir_del_roi_resetea_estado():
    tr = _track(1, (200, 200, 210, 210), in_roi_since=100.0, fired=True)
    r = evaluate_dwell([tr], _roi(), ts=120.0, dwell_sec=2.0, cooldown_sec=60.0, last_alert_ts=102.0)
    assert tr.in_roi_since is None
    assert not tr.fired
    assert r.dwell_map == {}
    assert r.triggered_track is None


def test_reentrada_tras_salir_vuelve_a_contar():
    tr = _track(1, (40, 40, 60, 60), in_roi_since=None, fired=False)
    r1 = evaluate_dwell([tr], _roi(), ts=200.0, dwell_sec=2.0, cooldown_sec=60.0, last_alert_ts=0.0)
    assert tr.in_roi_since == 200.0
    assert r1.dwell_map[1] == 0.0
    r2 = evaluate_dwell([tr], _roi(), ts=202.0, dwell_sec=2.0, cooldown_sec=60.0, last_alert_ts=0.0)
    assert r2.triggered_track is tr


def test_cooldown_bloquea_segundo_disparo():
    tr = _track(1, (40, 40, 60, 60), in_roi_since=100.0)
    r = evaluate_dwell([tr], _roi(), ts=103.0, dwell_sec=2.0, cooldown_sec=60.0, last_alert_ts=102.0)
    assert r.triggered_track is None
    assert not tr.fired


def test_cooldown_permite_disparo_al_vencer():
    tr = _track(1, (40, 40, 60, 60), in_roi_since=100.0)
    r = evaluate_dwell([tr], _roi(), ts=163.0, dwell_sec=2.0, cooldown_sec=60.0, last_alert_ts=102.0)
    assert r.triggered_track is tr
    assert r.new_alert_ts == 163.0


def test_un_solo_disparo_por_frame_con_multiples_tracks():
    t1 = _track(1, (40, 40, 60, 60), in_roi_since=100.0)
    t2 = _track(2, (70, 70, 90, 90), in_roi_since=100.0)
    r = evaluate_dwell([t1, t2], _roi(), ts=103.0, dwell_sec=2.0, cooldown_sec=60.0, last_alert_ts=0.0)
    assert r.triggered_track is not None
    # el primero dispara; el segundo queda bloqueado por cooldown en el mismo ts
    fired = [t for t in (t1, t2) if t.fired]
    assert len(fired) == 1


def test_inside_stats_cuenta_y_max_dwell():
    t1 = _track(1, (0, 0, 1, 1))
    t2 = _track(2, (0, 0, 1, 1))
    count, max_d = inside_stats([t1, t2], {1: 3.0, 2: 7.5})
    assert count == 2
    assert max_d == 7.5
    # dwell 0 no cuenta
    count2, max2 = inside_stats([t1, t2], {1: 0.0})
    assert count2 == 0
    assert max2 == 0.0


def test_track_to_overlay_serializa():
    tr = _track(5, (10.0, 20.0, 30.0, 40.0))
    out = track_to_overlay(tr, _roi(), {5: 1.234})
    assert out["id"] == 5
    assert out["bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert out["centroid"] == [20, 30]
    assert out["inside"] is True
    assert out["dwell_s"] == 1.23
