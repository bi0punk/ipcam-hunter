#!/usr/bin/env python3
import os
import time
import json
from pathlib import Path
from typing import Generator, Optional
from fastapi import HTTPException
import cv2
import numpy as np
import requests
import yaml
from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

APP_TITLE = "ROI Dashboard (RTSP + ROI + WhatsApp waha)"

def load_cfg(cfg_path: Path) -> dict:
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config no existe: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def roi_poly(cfg: dict) -> np.ndarray:
    pts = cfg["roi"]["points"]
    return np.array(pts, dtype=np.int32)

def draw_roi(img: np.ndarray, roi_pts: np.ndarray) -> np.ndarray:
    out = img.copy()
    cv2.polylines(out, [roi_pts.astype(np.int32)], True, (0,255,0), 2)
    cv2.putText(out, time.strftime("%Y-%m-%d %H:%M:%S"), (10,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return out

def mjpeg_generator(rtsp_url: str, process_width: int, fps_limit: float, roi_pts: np.ndarray) -> Generator[bytes, None, None]:
    cap = None
    backoff = 1.0
    next_frame = 0.0
    while True:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                time.sleep(backoff)
                continue

        ok, frame = cap.read()
        if not ok or frame is None:
            try:
                cap.release()
            except Exception:
                pass
            cap = None
            time.sleep(backoff)
            continue

        t = time.monotonic()
        if t < next_frame:
            time.sleep(max(0.0, next_frame - t))
        next_frame = time.monotonic() + (1.0 / max(1e-6, fps_limit))

        h, w = frame.shape[:2]
        scale = process_width / float(w)
        frame_p = cv2.resize(frame, (process_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        frame_p = draw_roi(frame_p, roi_pts)

        ok, jpg = cv2.imencode(".jpg", frame_p, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue
        data = jpg.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n" +
               data + b"\r\n")

def waha_headers() -> dict:
    api_key = os.environ.get("WAHA_API_KEY", "").strip()
    h = {"Content-Type": "application/json"}
    if api_key:
        h["X-Api-Key"] = api_key
    return h

def waha_base() -> str:
    return os.environ.get("WAHA_URL", "http://waha:3000").rstrip("/")

def status_path() -> Path:
    return Path(os.environ.get("STATUS_JSON", "/data/events/status.json"))

def events_dir() -> Path:
    return Path(os.environ.get("EVENTS_DIR", "/data/events"))

def safe_read_json(p: Path) -> Optional[dict]:
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None

app = FastAPI(title=APP_TITLE)
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    html = """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>__TITLE__</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:#0b1220; color:#e6eefc; margin:0; }
    header { padding:14px 18px; background:#0f1b33; border-bottom:1px solid #24324d; }
    .wrap { display:grid; grid-template-columns: 2fr 1fr; gap:14px; padding:14px; }
    .card { background:#0f1b33; border:1px solid #24324d; border-radius:12px; padding:12px; }
    .muted { color:#a9b7d6; font-size: 13px; }
    img { width:100%; border-radius:10px; border:1px solid #24324d; }
    code { background:#0b1220; padding:2px 6px; border-radius:6px; }
    button { padding:10px 12px; border-radius:10px; border:1px solid #24324d; background:#122242; color:#e6eefc; cursor:pointer; }
    button:hover { filter:brightness(1.1); }
    .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
    pre { white-space:pre-wrap; word-break:break-word; background:#0b1220; border:1px solid #24324d; border-radius:10px; padding:10px; }
    a { color:#8ab4ff; }
  </style>
</head>
<body>
<header>
  <div><b>ROI Dashboard</b> <span class="muted">RTSP + ROI overlay + estado + waha checks</span></div>
</header>

<div class="wrap">
  <div class="card">
    <div class="row" style="justify-content:space-between;">
      <div>
        <div><b>Live</b> <span class="muted">(MJPEG)</span></div>
        <div class="muted">Si no ves video, revisa RTSP/red/credenciales.</div>
      </div>
      <div class="row">
        <button onclick="refreshAll()">Refrescar</button>
        <a href="http://localhost:3000/" target="_blank">Swagger waha</a>
      </div>
    </div>
    <div style="margin-top:10px;">
      <img src="/mjpeg" alt="mjpeg stream"/>
    </div>
  </div>

  <div class="card">
    <div><b>Estado</b> <span class="muted">(worker → status.json)</span></div>
    <div class="row" style="margin-top:10px;">
      <button onclick="loadStatus()">Cargar estado</button>
      <button onclick="wahaHealth()">Waha health</button>
      <button onclick="listGroups()">Listar grupos</button>
    </div>
    <div style="margin-top:10px;">
      <pre id="out">Cargando…</pre>
    </div>
    <div style="margin-top:10px;">
      <div class="muted">Última evidencia (si existe):</div>
      <img id="lastEventImg" src="" alt="last event" style="display:none; margin-top:8px;"/>
    </div>
    <div style="margin-top:10px;" class="muted">
      <ul>
        <li>Si el navegador te pide user/pass en waha: prueba <code>any</code> / <code>WAHA_API_KEY</code>.</li>
      </ul>
    </div>
  </div>
</div>

<script>
async function loadStatus() {
  const r = await fetch('/api/status');
  const j = await r.json();
  document.getElementById('out').textContent = JSON.stringify(j, null, 2);

  const img = document.getElementById('lastEventImg');
  if (j && j.last_event_url) {
    img.src = j.last_event_url + '?t=' + Date.now();
    img.style.display = 'block';
  } else {
    img.style.display = 'none';
  }
}

async function wahaHealth() {
  const r = await fetch('/api/waha/health');
  const j = await r.json();
  document.getElementById('out').textContent = JSON.stringify(j, null, 2);
}

async function listGroups() {
  const r = await fetch('/api/waha/groups?session=default');
  const j = await r.json();
  document.getElementById('out').textContent = JSON.stringify(j, null, 2);
}

function refreshAll() { loadStatus(); }
loadStatus();
</script>

</body>
</html>
""".replace("__TITLE__", APP_TITLE)

    return HTMLResponse(html)

@app.get("/api/status")
def api_status():
    st = safe_read_json(status_path()) or {"status": "no_status_yet", "hint": "espera 2-3s; el worker escribe status.json"}
    last_event = st.get("last_event")
    if last_event:
        st["last_event_url"] = f"/events/{last_event}"
    return JSONResponse(st)

@app.get("/api/waha/health")
def api_waha_health():
    try:
        r = requests.get(f"{waha_base()}/api/sessions", headers=waha_headers(), timeout=10)
        body = None
        try:
            body = r.json() if r.content else None
        except Exception:
            body = r.text
        return JSONResponse({"ok": r.ok, "status_code": r.status_code, "body": body})
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.get("/api/waha/groups")
def api_waha_groups(session: str = "default", limit: int = 50, offset: int = 0):
    url = f"{waha_base()}/api/{session}/groups"
    try:
        r = requests.get(url, headers=waha_headers(), params={"limit": limit, "offset": offset}, timeout=15)
        body = None
        try:
            body = r.json() if r.content else None
        except Exception:
            body = r.text
        return JSONResponse({"ok": r.ok, "status_code": r.status_code, "body": body})
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.get("/events/{name}")
def get_event(name: str):
    p = events_dir() / name
    if not p.exists():
        raise HTTPException(status_code=404, detail="not found")
    data = p.read_bytes()
    return Response(content=data, media_type="image/jpeg")

@app.get("/mjpeg")
def mjpeg():
    cfg_path = Path(os.environ.get("CONFIG_PATH", "/app/config.yml"))
    cfg = load_cfg(cfg_path)
    

    cam = cfg.get("camera", {}) or {}
    rtsp = os.getenv("CAM_RTSP_URL", "").strip() or cam.get("rtsp_url") or cam.get("rtsp")

    if not rtsp:
        raise HTTPException(
            status_code=500,
        detail="Falta RTSP. Define CAM_RTSP_URL en .env (recomendado) o camera.rtsp_url en config.yaml"
    )
    process_w = int(cfg["camera"].get("process_width", 640))
    fps_limit = float(cfg["camera"].get("fps_limit", 8))
    poly = roi_poly(cfg)
    return StreamingResponse(
        mjpeg_generator(rtsp, process_w, fps_limit, poly),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )