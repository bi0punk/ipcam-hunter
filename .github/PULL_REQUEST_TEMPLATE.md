## Qué cambia

## Por qué

## Cómo se probó

- [ ] `ruff check .`
- [ ] `mypy app/roi_alert app/web/webapp scripts`
- [ ] `vulture app scripts vulture_whitelist.py --min-confidence 60`
- [ ] `pytest -q --cov=roi_alert --cov=webapp --cov-fail-under=65`
- [ ] `docker compose build` (si tocó Docker)

## Checklist

- [ ] Tests nuevos para el cambio
- [ ] Sin credenciales ni URLs reales en el diff
- [ ] `vulture_whitelist.py` actualizado si hay rutas/callbacks nuevos
- [ ] Docs actualizadas si aplica
