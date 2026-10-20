#!/usr/bin/env bash
# Запуск сервиса, Prometheus и Grafana в Docker Compose.
# Нужны артефакты services/models/model.cbm и services/assets/* (создаёт ноутбук 02_modeling).
set -euo pipefail

cd "$(dirname "$0")/../services"

docker compose up --build -d
echo "Сервис:     http://localhost:4601/docs"
echo "Метрики:    http://localhost:4601/metrics"
echo "Prometheus: http://localhost:9090"
echo "Grafana:    http://localhost:3000 (admin / admin)"
