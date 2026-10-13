import pytest

from roi_alert.tracker import IoUTracker, Track
from roi_alert.vision import iou


def _bb(x1, y1, x2, y2):
    return (float(x1), float(y1), float(x2), float(y2))


def test_crea_tracks_para_nuevas_detecciones():
    tr = IoUTracker(match_thres=0.5, ttl_sec=10.0)
    tracks = tr.update([_bb(0, 0, 10, 10)], ts=1.0)
    assert len(tracks) == 1
    assert tracks[0].tid == 1
    assert tracks[0].bbox == _bb(0, 0, 10, 10)
    assert tracks[0].last_seen == 1.0
    assert not tracks[0].fired
    assert tracks[0].in_roi_since is None


def test_ids_incrementales():
    tr = IoUTracker(match_thres=0.5, ttl_sec=10.0)
    t1 = tr.update([_bb(0, 0, 10, 10)], ts=1.0)[0]
    t2 = tr.update([_bb(100, 100, 110, 110)], ts=2.0)[1]
    assert t1.tid == 1
    assert t2.tid == 2


def test_matching_por_iou_preserva_tid_y_actualiza_bbox():
    tr = IoUTracker(match_thres=0.5, ttl_sec=10.0)
    tr.update([_bb(0, 0, 10, 10)], ts=1.0)
    # box ligeramente desplazado: IoU alto -> mismo track
    tracks = tr.update([_bb(1, 1, 11, 11)], ts=2.0)
    assert len(tracks) == 1
    assert tracks[0].tid == 1
    assert tracks[0].bbox == _bb(1, 1, 11, 11)
    assert tracks[0].last_seen == 2.0


def test_deteccion_bajo_threshold_es_track_nuevo():
    tr = IoUTracker(match_thres=0.5, ttl_sec=10.0)
    tr.update([_bb(0, 0, 10, 10)], ts=1.0)
    tracks = tr.update([_bb(50, 50, 60, 60)], ts=2.0)
    assert len(tracks) == 2
    assert [t.tid for t in tracks] == [1, 2]


def test_expected_ttl_elimina_tracks_viejos():
    tr = IoUTracker(match_thres=0.5, ttl_sec=5.0)
    tr.update([_bb(0, 0, 10, 10)], ts=1.0)
    assert len(tr.tracks) == 1
    # sin detección, tras superar el TTL el track se borra
    tracks = tr.update([], ts=10.0)
    assert len(tracks) == 0
    assert len(tr.tracks) == 0


def test_ttl_se_mide_desde_last_seen_no_desde_creacion():
    tr = IoUTracker(match_thres=0.5, ttl_sec=5.0)
    tr.update([_bb(0, 0, 30, 30)], ts=1.0)
    # detección mantiene vivo el track cada 4s
    for ts in (5.0, 9.0, 13.0):
        tracks = tr.update([_bb(1, 1, 31, 31)], ts=ts)
        assert len(tracks) == 1
    # 14s después de la última detección se paga el TTL
    tracks = tr.update([], ts=27.0)
    assert tracks == []


def test_sin_detecciones_no_crea_tracks():
    tr = IoUTracker(match_thres=0.5, ttl_sec=10.0)
    assert tr.update([], ts=0.0) == []


def test_match_threshold_exacto():
    # IoU exacto al umbral debe matchear (>=)
    bb = _bb(0, 0, 10, 10)
    tr = IoUTracker(match_thres=iou(bb, bb), ttl_sec=10.0)
    tr.update([bb], ts=1.0)
    tracks = tr.update([bb], ts=2.0)
    assert len(tracks) == 1


def test_varias_detecciones_asignan_cada_una_a_su_track():
    tr = IoUTracker(match_thres=0.5, ttl_sec=10.0)
    tr.update([_bb(0, 0, 10, 10), _bb(50, 50, 60, 60)], ts=1.0)
    # mismas boxes al siguiente frame -> mismos tids
    tr.update([_bb(1, 1, 11, 11), _bb(51, 51, 61, 61)], ts=2.0)
    for tr_ in (1, 2):
        assert any(t.tid == tr_ for t in tr.tracks.values())
    assert len(tr.tracks) == 2
    assert tr.next_id == 3


def test_track_es_un_dataclass_expuesto():
    tr = Track(tid=7, bbox=_bb(0, 0, 10, 10), last_seen=1.0)
    assert tr.tid == 7
    assert repr(tr).startswith("Track(tid=7")