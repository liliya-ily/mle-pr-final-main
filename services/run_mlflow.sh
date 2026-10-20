#!/usr/bin/env bash
# Запуск MLflow Tracking Server и Model Registry.
# Метаданные хранятся в SQLite (mlflow_server/mlflow.db), артефакты в папке mlflow_server/mlartifacts
# и отдаются клиентам самим сервером, поэтому ноутбуки логируют модели по HTTP без доступа к диску сервера.
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p mlflow_server/mlartifacts

mlflow server \
    --backend-store-uri "sqlite:///mlflow_server/mlflow.db" \
    --registry-store-uri "sqlite:///mlflow_server/mlflow.db" \
    --default-artifact-root "mlflow-artifacts:/" \
    --artifacts-destination "./mlflow_server/mlartifacts" \
    --serve-artifacts \
    --host 0.0.0.0 \
    --port "${MLFLOW_PORT:-5000}"
