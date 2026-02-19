import json
import time
from pathlib import Path


def now_ts() -> float:
    return time.monotonic()


def iso_time() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)