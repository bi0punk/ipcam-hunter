# Reporte de calidad — saneamiento inicial

Triage de la rama `dev` antes del refactor, con `ruff`, `mypy`, `vulture`,
`bandit` y `radon`, y su resolución.

## ruff (92 hallazgos → 0)

| Regla | Descripción | Resolución |
|---|---|---|
| `I001` | imports sin ordenar | `ruff --fix` |
| `UP035`/`UP006`/`UP045` | typing antiguo (`Dict`/`List`/`Tuple`/`Optional`) | `ruff --fix` → builtins y `X | None` |
| `F401` | imports sin usar (`os` en mjpeg) | eliminados |
| `BLE001` | `except Exception` ciego | intencional en workers; se dejó documentado |
| `B007` | variable de loop sin usar en tracker | renombrada a `_tid` |
| `E702` | sentencias con `;` en roi_json | separadas |
| `EXE001` | shebang sin permiso de ejecución | `chmod +x` |
| `RUF046` | `int()` redundante en create_roi | corregido |

## mypy (13 errores → 0)

- `evidence.py`: reasignación de `HTTPBasicAuth`/`HTTPDigestAuth` → anotado `AuthBase`.
- `waha.py`: `payload` inferido como `Collection[str]` → `dict[str, Any]`.
- `runner.py`: `Collection.append` al construir overlay → helper tipado.
- `ultralytics`: módulo sin `YOLO` (dir local de datos) → override
  `follow_imports = "skip"` en `pyproject.toml`.
- `webapp`: imports relativos “climbs too many namespaces” → `webapp` ahora es
  paquete regular (`__init__.py`).

## vulture

- `frame0` sin usar en runner → eliminado (se usa `_`).
- `settings.mjpeg_fps`/`mjpeg_boundary`/`live_path` sin uso → señalados (deuda
  del dashboard; no se tocó comportamiento).
- Falsos positivos: handlers de FastAPI y `stop()` de threads.

## bandit (6 B110 → 0)

Todos eran `try/except/pass` intencionales (dotenv opcional, `CAP_PROP_BUFFERSIZE`,
liberar `VideoCapture`). Se marcaron con `# nosec B110` documentando el motivo.
No se detectaron contraseñas hardcodeadas, `subprocess` ni `eval`/`exec`.

## radon — complejidad

| Función | Antes | Después |
|---|---|---|
| `runner.run` | **F (51)** | C (12) |
| `mjpeg.FrameHub._run` | C (20) | C (12) |
| `mjpeg.FrameHub._overlay` | C (19) | A (1) |

La bajada se logró extrayendo `dwell.py`, `events.py`, `settings.py`,
`overlay.py` y `streaming.py` (módulos pequeños y testeables).

## Seguridad

- `app/config.yaml` tenía `chat_id` real (`...@g.us`) e IP ISAPI hardcodeada.
  Ahora vacíos; se leen de `.env` (`WAHA_CHAT_ID`, `ISAPI_URL`).
- `evidence.mode` default cambiado a `rtsp` (seguro); `isapi` requiere
  credenciales por entorno.
- `tests/test_security.py` falla si reaparecen IDs/endpoints reales.
- Datos de runtime (`waha-sessions/*.sqlite3`, `ultralytics/*.json`) sacados
  del índice de git y añadidos a `.gitignore`.
