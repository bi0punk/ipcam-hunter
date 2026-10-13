# Migración de dependencias: requirements.txt → pyproject.toml

## Antes

Dos archivos de requisitos sin proyecto empaquetable:

- `app/requirements.txt` — worker (torch/torchvision/ultralytics, opencv, numpy,
  PyYAML, requests, python-dotenv) con índice CPU de PyTorch.
- `app/web/requirements.txt` — dashboard (fastapi, uvicorn, jinja2, opencv, numpy).

Sin `pyproject.toml`, sin extras, sin entrypoints, sin configuración de tools.

## Ahora

`pyproject.toml` es la fuente de verdad para desarrollo/CI:

| Grupo | Contenido |
|---|---|
| `dependencies` | numpy, opencv-python-headless, PyYAML, requests, python-dotenv, fastapi, uvicorn[standard], jinja2 |
| `[project.optional-dependencies].detect` | torch, torchvision, ultralytics (pesados; no se instalan por defecto) |
| `[project.optional-dependencies].dev` | pytest, pytest-mock, httpx, ruff, mypy, vulture, bandit, radon |

Extras instalables:

```bash
uv pip install -e ".[dev]"       # lint + tests (sin torch)
uv pip install -e ".[detect]"    # worker real (YOLO/PyTorch CPU)
```

Entrypoints (`console_scripts`):

- `roi-hunter` → `roi_dwell_alert:main`
- `roi-pick` → `roi_pick:main`

## Compatibilidad con Docker

Los `requirements.txt` **se mantienen** porque los Dockerfiles los usan:

- `Dockerfile` (worker) → `app/requirements.txt`
- `app/web/Dockerfile` (dashboard) → `app/web/requirements.txt`

Así el despliegue no cambia. Si quieres migrar Docker a pyproject:

```dockerfile
COPY pyproject.toml /app/pyproject.toml
COPY app /app
RUN pip install --no-cache-dir "/app[detect]"
```

## Tests y mocks

Como `torch`/`ultralytics` no se instalan en local/CI, `tests/conftest.py`
inyecta stubs de `ultralytics`, `torch` y `torchvision`. `cv2` (opencv) sí se
instala real. No hay dependencia de RTSP ni de red: se mockean con
`monkeypatch`/funciones inyectadas.
