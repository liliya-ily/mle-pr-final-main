#!/usr/bin/env bash
# Запуск MLflow Tracking Server и Model Registry.
# Доступы к S3 и PostgreSQL читаются из файла services/.env (шаблон: services/.env.example).
set -euo pipefail

cd "$(dirname "$0")"

set -a
source .env
set +a

: "${MLFLOW_S3_ENDPOINT_URL:?не задана в .env}"
: "${AWS_ACCESS_KEY_ID:?не задана в .env}"
: "${AWS_SECRET_ACCESS_KEY:?не задана в .env}"
: "${BUCKET:?не задана в .env}"
: "${DB_DESTINATION_HOST:?не задана в .env}"
: "${DB_DESTINATION_PORT:?не задана в .env}"
: "${DB_DESTINATION_NAME:?не задана в .env}"
: "${DB_DESTINATION_USER:?не задана в .env}"
: "${DB_DESTINATION_PASSWORD:?не задана в .env}"

DB_URI="postgresql://${DB_DESTINATION_USER}:${DB_DESTINATION_PASSWORD}@${DB_DESTINATION_HOST}:${DB_DESTINATION_PORT}/${DB_DESTINATION_NAME}"

mlflow server \
    --registry-store-uri "${DB_URI}" \
    --backend-store-uri "${DB_URI}" \
    --default-artifact-root "s3://${BUCKET}" \
    --host 0.0.0.0 \
    --port 5000 \
    --no-serve-artifacts
