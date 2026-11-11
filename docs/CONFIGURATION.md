# Configuración

Hay dos capas:

1. **`app/config.yaml`** — comportamiento del worker (cámara, modelo, ROI, dwell,
   storage, evidencia).
2. **Variables de entorno (`.env`)** — secretos y overrides (RTSP, WAHA, ISAPI,
   dashboard).

Las variables de entorno tienen prioridad sobre el YAML en todo lo sensible
(URLs y credenciales). Ninguna credencial real debe estar en el repositorio.

## Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `CAM_RTSP_URL` | vacío | Stream principal de la cámara |
| `CAM_RTSP_URL_SUB` | vacío | Substream; **tiene prioridad** sobre `CAM_RTSP_URL` |
| `CAM_RTSP_URL_PREVIEW` | vacío | Stream del dashboard MJPEG |
| `ISAPI_URL` | vacío | URL de snapshot HQ (Hikvision) |
| `ISAPI_AUTH` | `digest` | `digest` o `basic` |
| `ISAPI_USER` / `ISAPI_PASS` | vacío | Credenciales ISAPI |
| `WAHA_URL` | `http://waha:3000` | Base de WAHA |
| `WAHA_API_KEY` | vacío | API key de WAHA |
| `WAHA_SESSION` | `default` | Sesión de WhatsApp |
| `WAHA_CHAT_ID` | vacío | Grupo destino (`...@g.us`) |
| `EVENTS_DIR` | `/data/events` | Evidencias, `status.json`, `overlay.json` |
| `CONFIG_PATH` | `/app/config.yaml` | Ruta del YAML |
| `PREVIEW_WIDTH` | `640` | Ancho del stream del dashboard |
| `PREVIEW_FPS` | `12` | FPS de captura del dashboard |
| `JPEG_QUALITY` | `80` | Calidad JPEG del preview |
| `OVERLAY_JSON` | `/data/events/overlay.json` | Overlay que pinta el dashboard |
| `STATUS_JSON` | `/data/events/status.json` | Estado que pinta el dashboard |

## `app/config.yaml`

| Sección | Campo | Default | Descripción |
|---|---|---|---|
| `camera` | `process_width` | 640 | Ancho de inferencia |
| | `fps_limit` | 8 | FPS objetivo del loop |
| | `capture_fps_limit` | 20 | FPS de captura del grabber |
| | `reconnect_backoff_sec` | 2 | Espera entre reconexiones |
| | `detect_every_n_frames` | 1 | Saltar frames antes de inferir |
| | `overlay_hz` | 10 | Frecuencia de `overlay.json` |
| `model` | `yolo_weights` | `yolov8n.pt` | Pesos (se descargan si faltan) |
| | `conf_thres` | 0.35 | Umbral de confianza |
| `roi` | `name` | `zona_1` | Nombre mostrado en la alerta |
| | `points` | 4 puntos | Polígono del ROI en píxeles |
| | `from_json` / `json_path` | false | Cargar ROI desde `roi.json` |
| `logic` | `dwell_seconds` | 2.0 | Tiempo dentro del ROI para alertar |
| | `cooldown_seconds` | 60 | Anti-spam entre alertas |
| | `iou_match_thres` | 0.30 | Umbral del tracker |
| | `track_ttl_seconds` | 2.0 | Vida de un track sin detección |
| `storage` | `events_dir` | `./events` | Se redirige a `/data/events` si existe |
| | `save_annotated` | true | Guardar evidencia anotada |
| `whatsapp` | `session` | `default` | Sesión WAHA |
| | `chat_id` | vacío | Mejor definirlo por `WAHA_CHAT_ID` |
| | `caption_template` | … | Texto de la alerta |
| `evidence` | `mode` | `rtsp` | `rtsp` (frame) o `isapi` (HQ) |
| | `timeout_s` | 4 | Timeout del snapshot |
| | `fallback_to_rtsp` | true | Si ISAPI falla, usar frame |
| `isapi` | `url` / `auth` | vacío / digest | Mejor por entorno |

## Ejemplos

### Cámara Hikvision con substream y snapshot HQ

```env
CAM_RTSP_URL=rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101
CAM_RTSP_URL_SUB=rtsp://user:pass@192.168.1.64:554/Streaming/Channels/102
CAM_RTSP_URL_PREVIEW=rtsp://user:pass@192.168.1.64:554/Streaming/Channels/102
ISAPI_URL=http://192.168.1.64/ISAPI/Streaming/channels/101/picture
ISAPI_USER=user
ISAPI_PASS=pass
```

```yaml
evidence:
  mode: "isapi"
```

### Solo dashboard (sin worker)

```bash
uv venv .venv
uv pip install -r app/web/requirements.txt
EVENTS_DIR=./events uvicorn webapp.main:app --app-dir app/web --port 8081
```

### Rendimiento

- `fps_limit: 4` y `process_width: 480` para CPU de 1 núcleo.
- Usa `CAM_RTSP_URL_SUB`; el stream principal en 4K satura la decodificación.
- `detect_every_n_frames: 2` reduce inferencias sin perder dwell (el tracker
  mantiene el track entre frames).
