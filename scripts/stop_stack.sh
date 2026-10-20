#!/usr/bin/env bash
# Остановка сервиса, Prometheus и Grafana.
set -euo pipefail

cd "$(dirname "$0")/../services"
docker compose down
