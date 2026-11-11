# Despliegue

Tres piezas: **WAHA** (WhatsApp), **roi-alert** (worker YOLO) y **dashboard**
(FastAPI + MJPEG). El compose ya las orquesta.

> Las imágenes publicadas usan **Python 3.15**. Para el entorno local se recomienda 3.11/3.12.

## 1. Compose (recomendado)

```bash
git clone https://github.com/bi0punk/ipcam-hunter.git
cd ipcam-hunter
cp .example.env .env
# edita .env: CAM_RTSP_URL, WAHA_API_KEY, WAHA_CHAT_ID, ISAPI_*
sudo install -d -o 10001 -g 10001 events cache ultralytics
docker compose up -d
docker compose ps
```

- Dashboard: <http://localhost:8081>
- WAHA: <http://localhost:3000>

El worker y el dashboard corren como **uid 10001** (sin root). Los volúmenes
`events/`, `cache/` y `ultralytics/` deben pertenecer a ese uid:

```bash
sudo chown -R 10001:10001 events cache ultralytics
```

## 2. Imágenes publicadas (GHCR)

Cada push a `main` publica:

- `ghcr.io/bi0punk/ipcam-hunter:latest` (worker)
- `ghcr.io/bi0punk/ipcam-hunter-dashboard:latest` (dashboard)

Los tags `vX.Y.Z` generan `:X.Y.Z`.

```bash
docker login ghcr.io -u bi0punk
docker pull ghcr.io/bi0punk/ipcam-hunter:latest
docker pull ghcr.io/bi0punk/ipcam-hunter-dashboard:latest
```

Puedes sustituir `build:` por `image:` en el compose si no quieres compilar.

## 3. systemd (worker sin Docker)

```bash
sudo useradd -r -s /usr/sbin/nologin -u 10001 ipcam
sudo mkdir -p /opt/ipcam-hunter /var/lib/ipcam/events
sudo chown -R 10001:10001 /opt/ipcam-hunter /var/lib/ipcam
sudo -u ipcam git clone https://github.com/bi0punk/ipcam-hunter.git /opt/ipcam-hunter
cd /opt/ipcam-hunter
sudo -u ipcam python3.11 -m venv .venv
sudo -u ipcam .venv/bin/pip install -e ".[detect]"
sudo -u ipcam cp .example.env .env
```

Unidad (`/etc/systemd/system/ipcam-hunter.service`):

```ini
[Unit]
Description=ROI dwell alert (ipcam-hunter)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ipcam
Group=ipcam
WorkingDirectory=/opt/ipcam-hunter
EnvironmentFile=/opt/ipcam-hunter/.env
Environment=EVENTS_DIR=/var/lib/ipcam/events
ExecStart=/opt/ipcam-hunter/.venv/bin/python app/roi_dwell_alert.py --config /opt/ipcam-hunter/app/config.yaml
Restart=always
RestartSec=5
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/var/lib/ipcam/events

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ipcam-hunter
journalctl -u ipcam-hunter -f
```

## 4. WAHA / WhatsApp

1. Abre `http://TU_SERVIDOR:3000` y escanea el QR con el teléfono del número
   que enviará las alertas.
2. Define `WAHA_API_KEY` y `WAHA_CHAT_ID` (grupo `...@g.us`).
3. Prueba:

```bash
curl -s -H "X-Api-Key: $WAHA_API_KEY" http://localhost:3000/api/sessions
bash scripts/compose_urls.sh   # imprime dashboard y swagger
```

## 5. Actualización

```bash
git pull --ff-only
docker compose build && docker compose up -d
# systemd:
sudo -u ipcam /opt/ipcam-hunter/.venv/bin/pip install -e ".[detect]"
sudo systemctl restart ipcam-hunter
```

## 6. Checklist de producción

- [ ] `.env` completo y **fuera** de git.
- [ ] `events/`, `cache/`, `ultralytics/` con dueño 10001.
- [ ] `/health` del dashboard en verde.
- [ ] `status.json` actualizándose (worker healthy).
- [ ] Alerta de prueba recibida en el grupo de WhatsApp.
- [ ] Retención de evidencias (cron o limpieza manual).
- [ ] Cámara y WAHA en la misma red/VPN; nada expuesto a internet.
- [ ] `docker compose logs` revisado tras 24 h.
