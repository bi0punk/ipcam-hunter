# Troubleshooting

Orden de diagnóstico: **contenedores → salud → cámara → modelo → WAHA**.

## 1. Contenedores y salud

```bash
docker compose ps
docker inspect --format '{{.State.Health.Status}}' roi-alert roi-dashboard
docker compose logs --tail=100 roi-alert
curl -fsS http://localhost:8081/health
```

| Síntoma | Causa | Solución |
|---|---|---|
| `roi-alert` reinicia en bucle | `config.yaml` inválido | `python -c "import yaml;print(yaml.safe_load(open('app/config.yaml')))"` |
| `roi-alert` unhealthy | No escribe `status.json` | Revisa RTSP y logs del worker |
| `roi-dashboard` unhealthy | No responde `/health` | `docker compose logs dashboard` |
| `Permission denied` | Volumen de root | `sudo chown -R 10001:10001 events cache ultralytics` |
| Puerto 8081 ocupado | Otro servicio | Cambia el mapeo en compose |

## 2. Cámara / RTSP

```bash
ffplay -rtsp_transport tcp -i "rtsp://user:pass@IP:554/..."
```

| Causa | Solución |
|---|---|
| Stream principal saturado | Usa `CAM_RTSP_URL_SUB` |
| Credenciales con caracteres | Codifica la password |
| Timeout de red | Revisa VPN/VLAN y el firewall |
| Cámara limita conexiones | Cierra clientes extra |

## 3. Detección

- Confirma que el modelo se descargó (`yolov8n.pt`) en `./ultralytics` o el
  directorio de trabajo.
- Baja `model.conf_thres` a 0.25 y prueba.
- El ROI debe tener ≥ 3 puntos y estar en píxeles del frame procesado
  (`process_width`). Usa `bash scripts/create_roi.py` o `roi-pick`.
- Si detecta pero no alerta: `logic.dwell_seconds` (2 s) y `cooldown_seconds`
  (60 s) pueden estar filtrando.
- `status.json` muestra `tracks` y `max_dwell`; úsalo para ver si el tracker
  ve a la persona.

## 4. WhatsApp / WAHA

```bash
curl -s -H "X-Api-Key: $WAHA_API_KEY" http://localhost:3000/api/sessions
curl -s "http://localhost:8081/api/waha/health"
```

| Síntoma | Solución |
|---|---|
| Sesión `STOPPED` | Escanea el QR de nuevo |
| `WAHA_CHAT_ID` inválido | Debe terminar en `@g.us` |
| No llega la imagen | Revisa `waha-sessions/` y la API key |
| Error 401 | `WAHA_API_KEY` distinta entre worker y contenedor |

## 5. Rendimiento

- 1 núcleo: `fps_limit: 4`, `process_width: 480`, `detect_every_n_frames: 2`.
- Substream 720p antes que principal 4K.
- `overlay_hz` a 5 si el dashboard consume mucho.
- No ejecutes el dashboard y el worker en máquinas distintas sin compartir
  `events/` (NFS/SMB no recomendado para `status.json` en tiempo real).

## 6. CI en rojo

```bash
uv venv .venv --python 3.12
uv pip install -e ".[dev]"
.venv/bin/ruff check .
.venv/bin/mypy app/roi_alert app/web/webapp scripts
.venv/bin/vulture app scripts vulture_whitelist.py --min-confidence 60
.venv/bin/bandit -r app scripts -q
.venv/bin/pytest -q --cov=roi_alert --cov=webapp --cov-fail-under=65
```

Errores típicos:

- **vulture**: un símbolo nuevo usado solo desde un decorador de FastAPI debe
  añadirse a `vulture_whitelist.py`.
- **mypy**: `ultralytics` se ignora en `pyproject.toml`; no importes tipos de
  allí en el worker.
- **pytest**: los tests mockean torch/ultralytics; si un test nuevo importa
  `ultralytics` de verdad, usa el stub de `tests/conftest.py`.

## 7. Recolectar diagnóstico

```bash
docker compose ps > diag.txt
docker compose logs --tail=300 >> diag.txt
curl -s localhost:8081/api/status >> diag.txt
cat events/status.json >> diag.txt
```

Adjunta `diag.txt`, la versión del repo (`git rev-parse --short HEAD`) y el
modelo usado al abrir un issue.
