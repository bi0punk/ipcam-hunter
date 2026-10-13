import json
import time

from roi_alert.utils import iso_time, now_ts, safe_write_json


def test_iso_time_formato():
    s = iso_time()
    time.strptime(s, "%Y-%m-%d %H:%M:%S")  # lanza si no coincide


def test_now_ts_monotonic_mayor_cero():
    assert now_ts() > 0


def test_safe_write_json_escribe_contenido(tmp_path):
    target = tmp_path / "out.json"
    safe_write_json(target, {"a": 1, "b": [1, 2]})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": [1, 2]}


def test_safe_write_json_sin_tmp_residual(tmp_path):
    target = tmp_path / "out.json"
    safe_write_json(target, {"x": 1})
    assert not (tmp_path / "out.tmp").exists()
    assert target.exists()


def test_safe_write_json_sobrescribe(tmp_path):
    target = tmp_path / "out.json"
    safe_write_json(target, {"v": 1})
    safe_write_json(target, {"v": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 2}