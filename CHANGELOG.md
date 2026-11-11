# Changelog

Formato [Keep a Changelog](https://keepachangelog.com/es/1.1.0/) y
[SemVer](https://semver.org/lang/es/).

## [1.0.0] - 2026-11-11

Primera versión lista para desplegar y mantener.

### Añadido

- Endpoint `/health` en el dashboard para probes y `HEALTHCHECK` de Docker.
- Healthcheck del worker basado en la frescura de `events/status.json`.
- CI completo: calidad (ruff/mypy/vulture/bandit), matriz 3.11/3.12 con
  cobertura, `pip-audit` y build/publicación de ambas imágenes en GHCR.
- `vulture_whitelist.py` para que el chequeo de código muerto sea bloqueante.
- Dependabot (pip, actions, Docker), plantillas de issue y checklist de PR.
- Documentación: configuración, despliegue, troubleshooting, seguridad y
  contribución.
- `pytest-cov` y gate de cobertura del 65 % (cobertura actual: 69 %).

### Cambiado

- FastAPI 0.162 / Starlette 1.10 / Uvicorn 0.54 / Jinja2 3.1.6.
- Worker y dashboard corren como **uid 10001**, con `no-new-privileges`.
- Las cachés de Ultralytics y pip viven en `/home/hunter`, no en `/root`.
- Compose referencia las imágenes publicadas y declara healthchecks reales.

### Corregido

- Los tests de `resolve_events_dir` fallaban en Python 3.11 al subclasear
  `Path` sin `_flavour`; ahora se inyecta el predicado de existencia.
- Se eliminan `MJPEG_FPS`, `live_path`, `live_filename` y `mjpeg_boundary`,
  configuración que nadie consumía.
- El callback de OpenCV de `roi_pick` marca explícitamente sus argumentos no
  usados.

### Seguridad

- 8 CVEs de Starlette 0.38 resueltas al actualizar FastAPI.
- `requests>=2.33` y `python-dotenv>=1.2.2` para no reintroducir CVEs.
- `pip-audit` limpio y ejecutándose en cada push.

## [0.1.0] - 2026-10-12

- Detección de personas por ROI con YOLO sobre RTSP.
- Tracking IoU y alertas por dwell con cooldown.
- Evidencia anotada y snapshot ISAPI opcional.
- Envío a grupos de WhatsApp vía WAHA.
- Dashboard FastAPI con stream MJPEG y overlay.
