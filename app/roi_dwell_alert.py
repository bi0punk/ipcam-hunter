#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

# Soporte opcional: permite correr el script fuera de docker compose leyendo .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from roi_alert.runner import run  # noqa: E402


def main():
    default_cfg = os.getenv("CONFIG_PATH", "").strip() or "app/config.yaml"
    ap = argparse.ArgumentParser(description="ROI dwell alert (modular)")
    ap.add_argument("--config", default=default_cfg, help="Ruta a config.yaml")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    run(cfg_path)


if __name__ == "__main__":
    main()