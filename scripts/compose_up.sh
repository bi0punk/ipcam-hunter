#!/usr/bin/env bash
set -euo pipefail
if [ ! -f .env ]; then
  echo "No existe .env. Crea uno: cp .env.example .env"
  exit 1
fi
mkdir -p waha-sessions events cache ultralytics
docker compose up -d --build
echo "OK. Swagger waha: http://localhost:3000/"
