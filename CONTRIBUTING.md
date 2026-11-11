# Contribuir

Rama de trabajo: `dev`. `main` recibe merges `--no-ff` cuando el CI está verde.

## Entorno

```bash
git clone https://github.com/bi0punk/ipcam-hunter.git
cd ipcam-hunter
uv venv .venv --python 3.12
uv pip install -e ".[dev]"
```

El extra `[detect]` (torch/ultralytics) solo hace falta para ejecutar el worker
real; los tests lo mockean.

## Flujo

```
main  ←  merge --no-ff  ←  dev  ←  feat/* | fix/* | docs/*
```

1. Crea la rama desde `dev`.
2. Commits atómicos en imperativo (`feat`, `fix`, `refactor`, `test`, `docs`,
   `build`, `ci`, `chore`).
3. PR contra `dev` con el CI en verde.
4. Merge a `main` y publicación de imágenes.

## Calidad

```bash
ruff check .
mypy app/roi_alert app/web/webapp scripts
vulture app scripts vulture_whitelist.py --min-confidence 60
bandit -r app scripts -q
pytest -q --cov=roi_alert --cov=webapp --cov-fail-under=65
pip-audit
```

Todo corre en CI. Si agregas una ruta de FastAPI o un método de ciclo de vida,
añádelo a `vulture_whitelist.py` para que el chequeo siga siendo bloqueante.

## Tests

- `tests/test_*` — dominio del worker (cfg, dwell, tracker, visión, WAHA).
- `tests/web/` — dashboard (overlay, streaming, MJPEG, health).
- `tests/conftest.py` — stubs de `ultralytics`/`torch` y frames sintéticos.

No descargues modelos en tests: usa frames artificiales y los stubs.

## Estilo

- Python 3.11+, tipado, líneas de 120.
- `logging` en vez de `print` (excepto scripts interactivos como `roi_pick`).
- Los errores de red/cámara se registran y el worker reintenta; no tumban el
  proceso.

## Documentación

- `README.md` — visión general y uso.
- `docs/CONFIGURATION.md` — variables y `config.yaml`.
- `docs/DEPLOYMENT.md` — compose, GHCR, systemd.
- `docs/TROUBLESHOOTING.md` — diagnóstico por síntoma.
