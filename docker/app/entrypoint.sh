#!/usr/bin/env bash
set -euo pipefail
CONFIG_PATH="${CONFIG_PATH:-/app/config.yaml}"
echo "[entrypoint] Starting ROI dwell alert with config: $CONFIG_PATH"
python /app/roi_dwell_alert.py --config "$CONFIG_PATH"
