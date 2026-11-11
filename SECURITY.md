# Política de seguridad

## Versiones soportadas

| Versión | Soporte |
|---|---|
| 1.x | Sí (rama por defecto) |
| < 1.0 | No |

## Cómo reportar

Usa **GitHub Security Advisories** (Security → Report a vulnerability) o un
issue privado. No publiques detalles explotables.

Incluye: versión/commit, componente (worker, dashboard, WAHA, ISAPI, CI),
pasos de reproducción e impacto (exposición de credenciales, bypass del
dashboard, SSRF, DoS…). No adjuntes tokens ni contraseñas reales.

Respuesta objetivo: acuse en 72 h y parche o mitigación en 14 días.

## Alcance

Se consideran vulnerabilidades:

- exposición de `CAM_RTSP_URL`, `WAHA_API_KEY`, `ISAPI_PASS` o `.env`;
- dashboard accesible sin autenticación fuera de localhost;
- SSRF hacia la red interna vía URLs RTSP/ISAPI configurables;
- inyección de comandos en `scripts/` o en el entrypoint;
- lectura de evidencias (`events/`) por usuarios no autorizados;
- dependencias con CVE en el entorno de ejecución.

Fuera de alcance: acceso físico a la cámara o a la red, vulnerabilidades de
Ultralytics/OpenCV/WAHA sin vector propio, y despliegues que exponen el
dashboard a internet sin proxy ni auth.

## Recomendaciones de despliegue

- El dashboard escucha en `8081`; publícalo solo en localhost, VPN o detrás de
  un reverse proxy con TLS y auth.
- No expongas RTSP ni WAHA a internet.
- Rota `WAHA_API_KEY` si apareció en un log o commit.
- Ejecuta los contenedores sin root (ya usan uid 10001) y con
  `no-new-privileges`.
- Mantén dependencias al día: Dependabot + `pip-audit` en CI.

## Manejo de secretos

`.env` está en `.gitignore`. Si un secreto se filtró: rótalo en el origen,
elimínalo del historial (`git filter-repo`) y verifica logs/issues/artefactos.
