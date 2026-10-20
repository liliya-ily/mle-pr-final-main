"""Тесты обработчика запросов FastApiHandler: валидация входа и формирование рекомендаций."""
import json

import pandas as pd
import pytest
from catboost import CatBoostClassifier

from services.ml_service.fast_api_handler import FastApiHandler

MONTH = "2016-01"


@pytest.fixture(scope="module")
def handler(artifacts_dir):
    model = CatBoostClassifier()
    model.load_model(str(artifacts_dir / "models" / "model.cbm"))
    personal_als = pd.read_parquet(artifacts_dir / "assets" / "personal_als.parquet")
    user_features = pd.read_parquet(artifacts_dir / "assets" / "user_features.parquet")
    config = json.loads((artifacts_dir / "assets" / "inference_config.json").read_text(encoding="utf-8"))
    return FastApiHandler(model, personal_als, user_features, config)


def test_returns_top_k_sorted_by_score(handler):
    result = handler.handle({"user_id": 101, "month": MONTH, "k": 3})
    scores = [r["score"] for r in result["recommendations"]]
    assert result["user_id"] == 101 and result["month"] == MONTH
    assert len(scores) == 3
    assert scores == sorted(scores, reverse=True)


def test_default_k_comes_from_config(handler):
    result = handler.handle({"user_id": 101, "month": MONTH})
    assert len(result["recommendations"]) == 3  # top_k из inference_config.json


def test_k_larger_than_candidates_returns_all_candidates(handler):
    result = handler.handle({"user_id": 101, "month": MONTH, "k": 50})
    assert len(result["recommendations"]) == 4


def test_products_are_unique(handler):
    products = [r["product_id"] for r in handler.handle({"user_id": 102, "month": MONTH, "k": 4})["recommendations"]]
    assert len(products) == len(set(products))


def test_unknown_user_returns_empty_list(handler):
    result = handler.handle({"user_id": 999999, "month": MONTH, "k": 3})
    assert result["recommendations"] == []


def test_unknown_month_returns_empty_list(handler):
    assert handler.handle({"user_id": 101, "month": "2030-01", "k": 3})["recommendations"] == []


@pytest.mark.parametrize(
    "payload",
    [
        {"month": MONTH},                                # нет user_id
        {"user_id": 101},                                # нет month
        {"user_id": "abc", "month": MONTH},              # user_id не число
        {"user_id": 101, "month": "2016/01"},            # неверный формат месяца
        {"user_id": 101, "month": 201601},               # месяц не строкой
        {"user_id": 101, "month": MONTH, "k": 0},        # k не положительное
        {"user_id": 101, "month": MONTH, "k": "5"},      # k строкой
        "not a dict",                                    # тело не словарь
    ],
)
def test_invalid_payload_raises_value_error(handler, payload):
    with pytest.raises(ValueError):
        handler.handle(payload)
