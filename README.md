# ROI Dwell Alert → WhatsApp (waha) — Full Docker Compose

Regla:
- si cualquier persona permanece ≥ 2s dentro del ROI en RTSP
- si sale del ROI: resetea
- al cumplir: snapshot → envío a grupo WhatsApp (chatId ...@g.us) vía waha
- anti-spam: cooldown_seconds

## Arranque rápido

```bash
cp .env.example .env
nano .env   # define WAHA_API_KEY

cp app/config.docker.example.yaml app/config.yaml
nano app/config.yaml  # rtsp_url, whatsapp.chat_id (grupo), whatsapp.api_key (=WAHA_API_KEY)

bash scripts/compose_up.sh
```

Swagger waha:
- http://TU_SERVIDOR:3000/

## Login WhatsApp (una vez)
En Swagger:
1) POST /api/sessions/ con {"name":"default"}
2) POST /api/sessions/default/start
3) escanea el QR (endpoint de screenshot/qr en Swagger)

## Obtener el chat_id del grupo (...@g.us)
Dentro del contenedor:

```bash
export WAHA_API_KEY="$(grep WAHA_API_KEY .env | cut -d= -f2)"
docker exec -it roi-alert python /app/tools/get_groups.py --waha http://waha:3000 --session default --api-key "$WAHA_API_KEY"
```

Copia el id del grupo y ponlo en app/config.yaml -> whatsapp.chat_id.

## Logs
```bash
bash scripts/compose_logs.sh
```

## Outputs
- snapshots: ./events/
- sesión WhatsApp: ./waha-sessions/
- cache modelos: ./cache/ y ./ultralytics/


# Dashboard web (Live video + ROI + estado)
- Dashboard: http://localhost:8081
- waha Swagger: http://localhost:3000

El dashboard:
- muestra video RTSP (MJPEG) con overlay del ROI
- muestra estado desde `./events/status.json` (lo escribe el worker)
- permite checks rápidos a waha: `/api/waha/health` y `/api/waha/groups?session=default`

## Si no puedes entrar a waha (popup user/password)
Ese popup es Basic Auth. Prueba:
- Username: `any`
- Password: tu `WAHA_API_KEY` (del `.env`)
