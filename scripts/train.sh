#!/usr/bin/env bash
# Запускает MLflow, выполняет один ноутбук и останавливает MLflow.
#
#   bash scripts/train.sh 01_eda        # исследование данных
#   bash scripts/train.sh 02_modeling   # признаки, обучение, артефакты для сервиса
#
# Требуется файл data/train_ver2.csv. После 02_modeling в services/models и services/assets лежат артефакты сервиса.
set -euo pipefail

NOTEBOOK="${1:-}"
if [ -z "$NOTEBOOK" ]; then
    echo "Использование: bash scripts/train.sh 01_eda|02_modeling" >&2
    exit 1
fi

cd "$(dirname "$0")/.."

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"

if [ ! -f "${DATA_DIR:-data}/train_ver2.csv" ]; then
    echo "Не найден ${DATA_DIR:-data}/train_ver2.csv: скачайте данные (см. README)" >&2
    exit 1
fi

mkdir -p mlflow_server
bash services/run_mlflow.sh > mlflow_server/mlflow.log 2>&1 &
MLFLOW_PID=$!
trap 'kill "$MLFLOW_PID" 2>/dev/null || true' EXIT

# ждём, пока сервер MLflow начнёт отвечать
for _ in $(seq 1 60); do
    if python -c "import urllib.request; urllib.request.urlopen('${MLFLOW_TRACKING_URI}/health')" 2>/dev/null; then
        break
    fi
    sleep 2
done

cd notebooks
echo "Выполняется ${NOTEBOOK}.ipynb"
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 "${NOTEBOOK}.ipynb"
echo "Готово: ${NOTEBOOK}"
