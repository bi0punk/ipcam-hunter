"""Worker asíncrono de eventos + persistencia/evidencia.

Separa del bucle de `runner`: cola de eventos, guardado de JPEG (RTSP o
ISAPI), overlay anotado y envío WhatsApp. La parte de red (evidence/waha)
se inyecta para poder mockearla en tests.
"""
import json
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .dwell import track_to_overlay
from .evidence import fetch_isapi_snapshot
from .utils import iso_time, safe_write_json
from .vision import draw_overlay


@dataclass
class EventContext:
    events_dir: Path
    evidence_mode: str
    isapi_url: str
    isapi_user: str
    isapi_pass: str
    isapi_auth: str
    evidence_timeout_s: float
    fallback_to_rtsp: bool
    save_annotated: bool
    caption_template: str
    roi_points: np.ndarray
    chat_id: str
    waha: Any = field(default=None)
    overlay_path: Path | None = None

    @classmethod
    def from_parts(cls, **kw) -> "EventContext":
        kw.setdefault("isapi_user", "")
        kw.setdefault("isapi_pass", "")
        kw.setdefault("isapi_auth", "digest")
        kw.setdefault("evidence_timeout_s", 4.0)
        kw.setdefault("fallback_to_rtsp", True)
        kw.setdefault("save_annotated", True)
        kw.setdefault("waha", None)
        return cls(**kw)


def build_caption(template: str, roi: str, dwell: float, ts_iso: str) -> str:
    return template.format(roi=roi, dwell=dwell, ts=ts_iso)


def save_evidence(job: dict, ctx: EventContext) -> Path | None:
    """Persiste evidencia del evento y devuelve el JPEG a enviar (o None si skip)."""
    stamp = job["stamp"]
    roi = job["roi"]
    tid = job["tid"]
    frame_p = job["frame_p"]
    tracks = job["tracks"]
    dwell_map = job["dwell_map"]

    base = f"{stamp}_tid{tid}_{roi}"
    jpg_path = ctx.events_dir / f"{base}.jpg"
    annot_path = ctx.events_dir / f"{base}_annot.jpg"

    saved_ok = False
    skip_send = False
    drawn: np.ndarray | None = None

    if ctx.evidence_mode == "isapi":
        try:
            data = fetch_isapi_snapshot(
                ctx.isapi_url, ctx.isapi_user, ctx.isapi_pass, ctx.isapi_auth, ctx.evidence_timeout_s
            )
            jpg_path.write_bytes(data)
            saved_ok = True
            print(f"[evidence] isapi ok -> {jpg_path.name} ({len(data)} bytes)")
        except Exception as e:
            print(f"[evidence][warn] isapi fallo: {e}")
            if not ctx.fallback_to_rtsp:
                skip_send = True

    if (not saved_ok) and (not skip_send):
        out_img = frame_p
        if ctx.save_annotated:
            out_img = draw_overlay(frame_p, ctx.roi_points, tracks, dwell_map, iso_time)
            drawn = out_img  # reutilizar para el annot (evita segundo dibujo)
        cv2.imwrite(str(jpg_path), out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    if ctx.save_annotated:
        try:
            over = drawn if drawn is not None else draw_overlay(frame_p, ctx.roi_points, tracks, dwell_map, iso_time)
            cv2.imwrite(str(annot_path), over, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        except Exception as e:
            print(f"[annot][warn] no pude guardar overlay: {e}")

    if skip_send:
        print("[alert][warn] no evidencia y fallback_to_rtsp=false -> no envío")
        return None
    return jpg_path


def _worker_once(job: dict, ctx: EventContext) -> None:
    caption = build_caption(ctx.caption_template, job["roi"], job["dwell"], job["ts_iso"])
    jpg_path = save_evidence(job, ctx)
    if jpg_path is None:
        return
    resp = ctx.waha.send_image_b64(ctx.chat_id, jpg_path, caption)
    print(f"[alert] enviado {jpg_path.name} -> {ctx.chat_id} | resp={json.dumps(resp)[:200]}")


class EventWorker:
    """Cola + thread. `submit` nunca bloquea (dropo si cola llena)."""

    def __init__(self, ctx: EventContext, maxsize: int = 10):
        self.ctx = ctx
        self._q: queue.Queue[dict | None] = queue.Queue(maxsize=maxsize)
        self._thr: threading.Thread | None = None

    def start(self) -> None:
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()

    def stop(self) -> None:
        self._q.put(None)
        if self._thr:
            self._thr.join(timeout=2.0)

    def submit(self, job: dict) -> bool:
        try:
            self._q.put_nowait(job)
            return True
        except queue.Full:
            print("[worker][warn] cola llena, dropeo evento (evitar bloqueo)")
            return False

    def _run(self) -> None:
        while True:
            job = self._q.get()
            if job is None:
                self._q.task_done()
                return
            try:
                _worker_once(job, self.ctx)
            except Exception as e:
                print(f"[worker][error] {e}")
            finally:
                self._q.task_done()


def write_overlay_json(path: Path, payload: dict) -> None:
    try:
        safe_write_json(path, payload)
    except Exception as e:
        print(f"[overlay][error] {e}")


def write_status_json(path: Path, payload: dict) -> None:
    try:
        safe_write_json(path, payload)
    except Exception as e:
        print(f"[status][error] {e}")


def build_overlay_payload(
    ts_iso: str,
    roi_name: str,
    roi_points: np.ndarray,
    orig_shape: tuple[int, int] | None,
    proc_shape: tuple[int, int] | None,
    tracks: list,
    dwell_map: dict[int, float],
) -> dict:
    return {
        "ts": ts_iso,
        "roi": {"name": roi_name, "points": roi_points.tolist()},
        "frame": {
            "orig_hw": list(orig_shape) if orig_shape else None,
            "proc_hw": list(proc_shape) if proc_shape else None,
        },
        "tracks": [track_to_overlay(tr, roi_points, dwell_map) for tr in tracks],
    }
