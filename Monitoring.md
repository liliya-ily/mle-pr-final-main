# Мониторинг рекомендательного сервиса

Мониторинг предназначен для FastAPI-сервиса рекомендаций банковских продуктов.

Prometheus собирает метрики с эндпоинта `/metrics`.

Сервис предоставляет:

- `/recommend` — получение рекомендаций
- `/metrics` — метрики Prometheus

Запуск сервисов:

```
cd services
docker compose up --build
```

Адреса:

- API: http://localhost:4601/recommend
- Метрики: http://localhost:4601/metrics
- Prometheus: http://localhost:9090

---

# Слои мониторинга

Мониторинг покрывает три уровня:

1. Инфраструктурный
2. Трафик и ошибки
3. Прикладной (бизнес-уровень)

---

## 1. Инфраструктурный уровень

### Метрики Python-процесса

Prometheus автоматически собирает:

- process_virtual_memory_bytes
- process_resident_memory_bytes
- python_gc_*

Назначение:

- контроль потребления памяти
- выявление утечек
- анализ нагрузки на процесс

Проверка:
```
curl -s http://localhost:4601/metrics | grep process_virtual_memory_bytes
```
---

## 2. Метрики трафика и ошибок

### Метрика: recommend_requests_total{status=...}

Счётчик запросов к /recommend.

Возможные значения status:

- ok
- validation_error
- model_error
- load_error

Метрика инкрементируется из кода сервиса при каждом обращении.

Проверка:
```
curl -s http://localhost:4601/metrics | grep recommend_requests_total
```
### Проверка статусов

Успешный запрос:
```
curl -X POST http://localhost:4601/recommend   -H "Content-Type: application/json"   -d '{"user_id": 15896, "month": "2016-01", "k": 5}'
```
Ошибка валидации:
```
curl -X POST http://localhost:4601/recommend   -H "Content-Type: application/json"   -d '{"user_id": "abc", "month": "2016-01"}'
```
Во втором случае сервис возвращает HTTP 400 и увеличивает счётчик validation_error.

---

## 3. Метрики реального времени (HTTP)

Через prometheus_fastapi_instrumentator автоматически собираются:

- http_requests_total
- http_request_duration_seconds
- http_response_size_bytes

Назначение:

- контроль RPS
- анализ latency
- контроль статусов HTTP

Пример запроса в Prometheus:
```
sum(rate(http_requests_total[1m])) by (status)
```
---

## Генерация нагрузки

Для проверки мониторинга используется скрипт:
```
python services/test_load.py
```
Скрипт:

- отправляет корректные запросы
- отправляет некорректный запрос (валидация)
- создаёт кратковременную нагрузку

После выполнения метрика recommend_requests_total увеличивается.
