#!/usr/bin/env bash
# Быстрая проверка запущенного сервиса: корректный запрос, запрос с ошибкой и метрики.
set -euo pipefail

URL="${SERVICE_URL:-http://localhost:4601}"

echo "Корректный запрос:"
code=$(curl -s -o /tmp/recommend_ok.json -w "%{http_code}" -X POST "$URL/recommend" \
    -H "Content-Type: application/json" \
    -d '{"user_id": 15889, "month": "2016-01", "k": 5}')
cat /tmp/recommend_ok.json; echo
[ "$code" = "200" ] || { echo "ожидался код 200, получен $code" >&2; exit 1; }

echo "Запрос с ошибкой (user_id строкой):"
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$URL/recommend" \
    -H "Content-Type: application/json" \
    -d '{"user_id": "abc", "month": "2016-01", "k": 5}')
[ "$code" = "400" ] || { echo "ожидался код 400, получен $code" >&2; exit 1; }
echo "код $code"

echo "Метрики:"
curl -s "$URL/metrics" | grep -E "^recommend_requests_total" || { echo "нет метрики recommend_requests_total" >&2; exit 1; }
