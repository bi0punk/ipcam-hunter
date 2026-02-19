import os
import time
import threading
import json
from pathlib import Path
from typing import Optional

import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

PORT = int(os.getenv("PORT", "8081"))

EVENTS_DIR = Path(os.getenv("EVENTS_DIR", "/data/events"))
OVERLAY_JSON = Path(os.getenv("OVERLAY_JSON", str(EVENTS_DIR / "overlay.json")))
STATUS_JSON = Path(os.getenv("STATUS_JSON", str(EVENTS_DIR / "status.json")))

RTSP = (os.getenv("CAM_RTSP_URL_PREVIEW", "").strip()
        or os.getenv("CAM_RTSP_URL", "").strip())

PREVIEW_WIDTH = int(os.getenv("PREVIEW_WIDTH", "640"))
PREVIEW_FPS = float(os.getenv("PREVIEW_FPS", "15"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))

if not RTSP:
    raise SystemExit("Falta CAM_RTSP_URL_PREVIEW o CAM_RTSP_URL para el dashboard")

app = FastAPI()


HTML = """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ROI Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg:        #080c10;
      --surface:   #0d1520;
      --surface2:  #111d2b;
      --border:    #1a2d42;
      --accent:    #00e5ff;
      --accent2:   #00ff8c;
      --warn:      #ffd166;
      --danger:    #ff4f6b;
      --text:      #c8dff0;
      --text-dim:  #5a7a96;
      --radius:    10px;
      --mono:      'JetBrains Mono', monospace;
      --sans:      'Syne', sans-serif;
    }

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: var(--sans);
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      overflow-x: hidden;
    }

    /* Subtle scanline overlay */
    body::before {
      content: '';
      position: fixed; inset: 0;
      background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,229,255,0.015) 2px,
        rgba(0,229,255,0.015) 4px
      );
      pointer-events: none;
      z-index: 9999;
    }

    /* ── Header ── */
    header {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 14px 20px;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(90deg, #0d1520 0%, #080c10 100%);
    }

    .logo-dot {
      width: 10px; height: 10px;
      border-radius: 50%;
      background: var(--accent2);
      box-shadow: 0 0 10px var(--accent2);
      animation: pulse 2s ease-in-out infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; box-shadow: 0 0 8px var(--accent2); }
      50%       { opacity: 0.5; box-shadow: 0 0 2px var(--accent2); }
    }

    header h1 {
      font-size: 16px;
      font-weight: 800;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--accent);
    }

    .header-sub {
      margin-left: auto;
      font-family: var(--mono);
      font-size: 11px;
      color: var(--text-dim);
    }

    #conn-badge {
      margin-left: 8px;
      padding: 2px 8px;
      border-radius: 20px;
      font-size: 10px;
      font-weight: 600;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      background: rgba(255,79,107,0.15);
      color: var(--danger);
      border: 1px solid rgba(255,79,107,0.3);
      transition: all .3s;
    }
    #conn-badge.live {
      background: rgba(0,255,140,0.12);
      color: var(--accent2);
      border-color: rgba(0,255,140,0.3);
    }

    /* ── Layout ── */
    .wrap {
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 16px;
      padding: 16px;
      max-width: 1400px;
    }

    @media (max-width: 900px) {
      .wrap { grid-template-columns: 1fr; }
    }

    .panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
    }

    .panel-header {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 10px 14px;
      border-bottom: 1px solid var(--border);
      background: var(--surface2);
    }

    .panel-header span {
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--text-dim);
    }

    .panel-icon {
      width: 6px; height: 6px;
      border-radius: 50%;
      background: var(--accent);
    }

    /* ── Video ── */
    .video-wrap {
      position: relative;
      display: block;
      background: #000;
      line-height: 0;
    }

    #vid {
      display: block;
      width: 100%;
      height: auto;
    }

    #ovl {
      position: absolute;
      left: 0; top: 0;
      pointer-events: none;
      width: 100%;
      height: 100%;
    }

    /* ── Side panel ── */
    .side { display: flex; flex-direction: column; gap: 16px; }

    /* Stat cards */
    .stats-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      padding: 12px;
    }

    .stat-card {
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
    }

    .stat-label {
      font-size: 9px;
      font-weight: 600;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--text-dim);
      margin-bottom: 4px;
    }

    .stat-value {
      font-family: var(--mono);
      font-size: 20px;
      font-weight: 600;
      color: var(--accent);
      line-height: 1;
    }

    .stat-value.ok   { color: var(--accent2); }
    .stat-value.warn { color: var(--warn); }
    .stat-value.err  { color: var(--danger); }

    /* JSON sections */
    .json-section {
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .json-section h3 {
      font-size: 10px;
      font-weight: 600;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--text-dim);
    }

    pre {
      font-family: var(--mono);
      font-size: 11px;
      line-height: 1.6;
      color: #7eb8d4;
      background: #060b11;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px;
      max-height: 220px;
      overflow-y: auto;
      white-space: pre-wrap;
      word-break: break-all;
    }

    pre::-webkit-scrollbar { width: 4px; }
    pre::-webkit-scrollbar-track { background: transparent; }
    pre::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

    /* Track list */
    .tracks-list {
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .track-item {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 8px 10px;
      background: var(--surface2);
      border-radius: 6px;
      border-left: 3px solid var(--warn);
      font-family: var(--mono);
      font-size: 11px;
      animation: fadeSlide .3s ease;
    }

    .track-item.inside { border-left-color: var(--accent); }

    @keyframes fadeSlide {
      from { opacity: 0; transform: translateX(-6px); }
      to   { opacity: 1; transform: none; }
    }

    .track-id {
      font-weight: 600;
      color: var(--accent);
      min-width: 36px;
    }

    .track-dwell {
      color: var(--text-dim);
      font-size: 10px;
    }

    .track-badge {
      margin-left: auto;
      font-size: 9px;
      font-weight: 600;
      letter-spacing: .06em;
      text-transform: uppercase;
      padding: 2px 6px;
      border-radius: 4px;
    }

    .inside .track-badge {
      background: rgba(0,229,255,0.12);
      color: var(--accent);
    }

    .outside .track-badge {
      background: rgba(255,209,102,0.12);
      color: var(--warn);
    }

    .no-tracks {
      font-family: var(--mono);
      font-size: 11px;
      color: var(--text-dim);
      padding: 8px 0;
    }

    /* Footer */
    footer {
      padding: 10px 20px;
      border-top: 1px solid var(--border);
      font-family: var(--mono);
      font-size: 10px;
      color: var(--text-dim);
      display: flex;
      gap: 20px;
    }

    .fps-bar {
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .fps-val { color: var(--accent); }
  </style>
</head>
<body>

<header>
  <div class="logo-dot"></div>
  <h1>ROI&nbsp;Dashboard</h1>
  <div class="header-sub">
    <span id="clock">--:--:--</span>
    <span id="conn-badge">offline</span>
  </div>
</header>

<div class="wrap">

  <!-- VIDEO -->
  <div class="panel">
    <div class="panel-header">
      <div class="panel-icon"></div>
      <span>Live Feed</span>
    </div>
    <div class="video-wrap">
      <img id="vid" src="/mjpeg" alt="stream" />
      <canvas id="ovl"></canvas>
    </div>
  </div>

  <!-- SIDE -->
  <div class="side">

    <!-- Stats cards -->
    <div class="panel">
      <div class="panel-header">
        <div class="panel-icon"></div>
        <span>Métricas</span>
      </div>
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-label">Dentro ROI</div>
          <div class="stat-value ok" id="s-inside">—</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Total tracks</div>
          <div class="stat-value" id="s-total">—</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Estado</div>
          <div class="stat-value ok" id="s-state">—</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Actualizando</div>
          <div class="stat-value" id="s-hz">10 Hz</div>
        </div>
      </div>
    </div>

    <!-- Tracks list -->
    <div class="panel">
      <div class="panel-header">
        <div class="panel-icon" style="background:var(--accent2)"></div>
        <span>Tracks activos</span>
      </div>
      <div class="tracks-list" id="tracks-list">
        <span class="no-tracks">Sin datos…</span>
      </div>
    </div>

    <!-- Raw JSON -->
    <div class="panel">
      <div class="panel-header">
        <div class="panel-icon" style="background:var(--text-dim)"></div>
        <span>JSON Estado</span>
      </div>
      <div class="json-section">
        <pre id="status">cargando…</pre>
      </div>
    </div>

    <div class="panel">
      <div class="panel-header">
        <div class="panel-icon" style="background:var(--text-dim)"></div>
        <span>JSON Overlay</span>
      </div>
      <div class="json-section">
        <pre id="overlay">cargando…</pre>
      </div>
    </div>

  </div>
</div>

<footer>
  <span>ROI Dashboard v2</span>
  <span class="fps-bar">Stream: <span class="fps-val" id="f-fps">—</span> fps</span>
  <span id="f-last">última actualización: —</span>
</footer>

<script>
const vid    = document.getElementById("vid");
const canvas = document.getElementById("ovl");
const ctx    = canvas.getContext("2d");

const statusEl  = document.getElementById("status");
const overlayEl = document.getElementById("overlay");
const tracksList = document.getElementById("tracks-list");
const connBadge  = document.getElementById("conn-badge");
const clockEl    = document.getElementById("clock");

// Stat elements
const sInside = document.getElementById("s-inside");
const sTotal  = document.getElementById("s-total");
const sState  = document.getElementById("s-state");
const fLast   = document.getElementById("f-last");

// ── Clock ──────────────────────────────────────────────
function tickClock() {
  clockEl.textContent = new Date().toLocaleTimeString("es-CL", { hour12: false });
}
setInterval(tickClock, 1000);
tickClock();

// ── Canvas sync ────────────────────────────────────────
function syncCanvas() {
  const w = vid.naturalWidth || vid.width;
  const h = vid.naturalHeight || vid.height;
  if (w && h) {
    canvas.width = w;
    canvas.height = h;
  }
}
vid.onload = syncCanvas;

// ── Draw overlay ───────────────────────────────────────
function drawOverlay(data) {
  if (!data || !data.roi) return;
  syncCanvas();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // ROI polygon
  const roi = data.roi;
  if (roi && roi.points && roi.points.length > 1) {
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#00ff8c";
    ctx.shadowColor  = "#00ff8c";
    ctx.shadowBlur   = 6;
    ctx.beginPath();
    ctx.moveTo(roi.points[0][0], roi.points[0][1]);
    for (let i = 1; i < roi.points.length; i++)
      ctx.lineTo(roi.points[i][0], roi.points[i][1]);
    ctx.closePath();
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  // Bounding boxes
  const tracks = data.tracks || [];
  for (const t of tracks) {
    const [x1, y1, x2, y2] = t.bbox;
    const color = t.inside ? "#00e5ff" : "#ffd166";
    ctx.lineWidth    = 2;
    ctx.strokeStyle  = color;
    ctx.shadowColor  = color;
    ctx.shadowBlur   = 8;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.shadowBlur   = 0;

    // Label
    const label = `id=${t.id}  ${t.dwell_s}s`;
    ctx.font = "bold 12px 'JetBrains Mono', monospace";
    const tw = ctx.measureText(label).width;
    const lx = x1, ly = Math.max(0, y1 - 18);

    ctx.fillStyle = "rgba(6,11,17,0.75)";
    ctx.fillRect(lx - 1, ly, tw + 10, 17);

    ctx.fillStyle = color;
    ctx.fillText(label, lx + 4, ly + 12);
  }
}

// ── Update stats cards ─────────────────────────────────
function updateStats(status, overlay) {
  const tracks = (overlay && overlay.tracks) ? overlay.tracks : [];
  const inside = tracks.filter(t => t.inside).length;

  sInside.textContent = inside;
  sTotal.textContent  = tracks.length;

  if (status && status.ok !== undefined) {
    sState.textContent = status.ok ? "OK" : "ERR";
    sState.className   = "stat-value " + (status.ok ? "ok" : "err");
  } else {
    sState.textContent = "—";
    sState.className   = "stat-value";
  }

  fLast.textContent = "última actualización: " + new Date().toLocaleTimeString("es-CL");
}

// ── Update tracks list ─────────────────────────────────
function updateTracksList(overlay) {
  const tracks = (overlay && overlay.tracks) ? overlay.tracks : [];
  if (!tracks.length) {
    tracksList.innerHTML = '<span class="no-tracks">Sin tracks activos</span>';
    return;
  }
  tracksList.innerHTML = tracks.map(t => `
    <div class="track-item ${t.inside ? 'inside' : 'outside'}">
      <span class="track-id">#${t.id}</span>
      <span class="track-dwell">${t.dwell_s}s dwell</span>
      <span class="track-badge">${t.inside ? 'dentro' : 'fuera'}</span>
    </div>
  `).join('');
}

// ── Poll loop ──────────────────────────────────────────
let pollErrors = 0;

async function poll() {
  try {
    const [s, o] = await Promise.all([
      fetch("/status").then(r => r.json()),
      fetch("/overlay").then(r => r.json())
    ]);

    statusEl.textContent  = JSON.stringify(s, null, 2);
    overlayEl.textContent = JSON.stringify(o, null, 2);

    drawOverlay(o);
    updateStats(s, o);
    updateTracksList(o);

    pollErrors = 0;
    connBadge.textContent = "live";
    connBadge.className   = "live";
  } catch (e) {
    pollErrors++;
    statusEl.textContent = "error: " + e;
    if (pollErrors > 3) {
      connBadge.textContent = "offline";
      connBadge.className   = "";
    }
  }
  setTimeout(poll, 100);
}

poll();
</script>
</body>
</html>
"""


class LatestFrame:
    def __init__(self, rtsp: str, width: int, fps: float):
        self.rtsp = rtsp
        self.width = width
        self.fps = max(1.0, fps)
        self.lock = threading.Lock()
        self.frame = None
        self.stop = False
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.backoff = 2.0

    def start(self):
        self.thread.start()

    def run(self):
        cap = None
        next_t = 0.0
        while not self.stop:
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(self.rtsp)
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                if not cap.isOpened():
                    time.sleep(self.backoff)
                    continue

            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                cap = None
                time.sleep(self.backoff)
                continue

            t = time.monotonic()
            if t < next_t:
                time.sleep(max(0.0, next_t - t))
            next_t = time.monotonic() + (1.0 / self.fps)

            h, w = frame.shape[:2]
            scale = self.width / float(w)
            frame_p = cv2.resize(
                frame,
                (self.width, int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
            with self.lock:
                self.frame = frame_p

    def get(self) -> Optional[bytes]:
        with self.lock:
            if self.frame is None:
                return None
            frame = self.frame.copy()

        ok, jpg = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        if not ok:
            return None
        return jpg.tobytes()


latest = LatestFrame(RTSP, PREVIEW_WIDTH, PREVIEW_FPS)
latest.start()


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/status")
def status():
    if STATUS_JSON.exists():
        try:
            return JSONResponse(json.loads(STATUS_JSON.read_text(encoding="utf-8")))
        except Exception:
            pass
    return JSONResponse({"ok": False, "msg": "status.json no disponible aún"})


@app.get("/overlay")
def overlay():
    if OVERLAY_JSON.exists():
        try:
            return JSONResponse(json.loads(OVERLAY_JSON.read_text(encoding="utf-8")))
        except Exception:
        
            pass
    return JSONResponse({"ok": False, "msg": "overlay.json no disponible aún"})


@app.get("/mjpeg")
def mjpeg():
    def gen():
        while True:
            data = latest.get()
            if data is None:
                time.sleep(0.05)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n" +
                data + b"\r\n"
            )
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")