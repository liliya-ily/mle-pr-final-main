"""Тесты HTTP-сервиса: загрузка артефактов при старте, эндпоинт /recommend и метрики Prometheus."""
import re

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(artifacts_dir):
    # корректируем путь к артефактам до импорта приложения
    import os

    os.environ["ARTIFACTS_DIR"] = str(artifacts_dir)
    from services.ml_service.main_monitoring import app

    with TestClient(app) as test_client:  # with запускает lifespan: загрузку артефактов
        yield test_client


def counter(client, status):
    """Текущее значение счётчика recommend_requests_total для статуса."""
    text = client.get("/metrics").text
    match = re.search(rf'recommend_requests_total\{{status="{status}"\}} ([0-9.e+]+)', text)
    return float(match.group(1)) if match else 0.0


def test_recommend_ok(client):
    before = counter(client, "ok")
    response = client.post("/recommend", json={"user_id": 101, "month": "2016-01", "k": 2})
    assert response.status_code == 200
    body = response.json()
    assert len(body["recommendations"]) == 2
    assert {"product_id", "score"} <= set(body["recommendations"][0])
    assert counter(client, "ok") == before + 1


def test_validation_error_returns_400_and_counts(client):
    before = counter(client, "validation_error")
    response = client.post("/recommend", json={"user_id": "abc", "month": "2016-01", "k": 2})
    assert response.status_code == 400
    assert "user_id" in response.json()["detail"]
    assert counter(client, "validation_error") == before + 1


def test_unknown_user_returns_empty_recommendations(client):
    response = client.post("/recommend", json={"user_id": 999999, "month": "2016-01", "k": 3})
    assert response.status_code == 200
    assert response.json()["recommendations"] == []


def test_metrics_endpoint_exposes_service_metrics(client):
    text = client.get("/metrics").text
    assert "recommend_requests_total" in text
    assert "http_request_duration_seconds" in text
