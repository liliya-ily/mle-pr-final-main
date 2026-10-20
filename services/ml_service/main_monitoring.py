# services/ml_service/main_monitoring.py

import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager

import pandas as pd
import s3fs
from catboost import CatBoostClassifier
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, Body, HTTPException

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter

from .fast_api_handler import FastApiHandler

logger = logging.getLogger("uvicorn.error")


def _load_env() -> None:
    """
    Загружает переменные окружения из .env.
    """
    load_dotenv(find_dotenv(), override=True)


def _storage_options() -> dict:
    """
    Формирует опции доступа к S3 для pandas.read_parquet и s3fs.
    """
    return {
        "key": os.getenv("AWS_ACCESS_KEY_ID"),
        "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "client_kwargs": {
            "endpoint_url": os.getenv("MLFLOW_S3_ENDPOINT_URL"),
            "region_name": os.getenv("AWS_DEFAULT_REGION"),
        },
    }


def _s3_base_path() -> str:
    """
    Базовый путь в S3, где лежат артефакты.
    """
    bucket = os.getenv("S3_BUCKET_NAME") or os.getenv("BUCKET")
    if not bucket:
        raise RuntimeError("S3 bucket env var not set (S3_BUCKET_NAME or BUCKET)")

    # фиксированный префикс как в ноутбуке
    return f"s3://{bucket}/bank_recsys_artifacts/"


def _download_catboost_model(fs: s3fs.S3FileSystem, s3_uri: str) -> CatBoostClassifier:
    """
    Модель скачивается во временный файл, затем загружается CatBoost'ом.
    """
    tmp_dir = tempfile.mkdtemp(prefix="cb_")
    local_path = os.path.join(tmp_dir, "model.cbm")

    fs.get(s3_uri, local_path)

    model = CatBoostClassifier()
    model.load_model(local_path)
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan FastAPI:
    - при старте: загрузка артефактов из S3 в память и создание handler
    - при остановке: ничего специального не требуется
    """
    _load_env()
    storage_opts = _storage_options()

    base = _s3_base_path()
    model_uri = base + "model.cbm"
    personal_als_uri = base + "personal_als.parquet"
    user_features_uri = base + "user_features.parquet"
    cfg_uri = base + "inference_config.json"

    logger.info("Loading artifacts from S3 base: %s", base)

    fs = s3fs.S3FileSystem(**storage_opts)

    # config
    with fs.open(cfg_uri, "rb") as f:
        infer_cfg = json.loads(f.read().decode("utf-8"))

    # data
    personal_als = pd.read_parquet(personal_als_uri, storage_options=storage_opts)
    user_features = pd.read_parquet(user_features_uri, storage_options=storage_opts)

    # нормализация типов для join/фильтров
    personal_als["month_str"] = personal_als["month_str"].astype(str)
    user_features["month_str"] = user_features["month_str"].astype(str)

    # model
    model = _download_catboost_model(fs=fs, s3_uri=model_uri)

    app.state.handler = FastApiHandler(
        model=model,
        personal_als=personal_als,
        user_features=user_features,
        infer_cfg=infer_cfg,
        id_col="ncodpers",
        month_col="month_str",
    )

    logger.info("Artifacts loaded: personal_als=%s rows, user_features=%s rows", len(personal_als), len(user_features))
    logger.info("Service ready")
    yield
    logger.info("Service stopping")


app = FastAPI(title="bank_recsys", lifespan=lifespan)

# Автоматические метрики HTTP + /metrics
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Кастомный счётчик запросов /recommend по статусам
requests_counter = Counter(
    "recommend_requests_total",
    "Total requests processed by /recommend endpoint",
    ["status"],
)


@app.post("/recommend")
def recommend(payload: dict = Body(..., example={"user_id": 15896, "month": "2016-01", "k": 5})):
    """
    Возвращает рекомендации.

    Ошибки валидации должны быть 400, чтобы это было видно и в тестах, и в метриках.
    """
    try:
        result = app.state.handler.handle(payload)
        requests_counter.labels(status="ok").inc()
        return result

    except ValueError as e:
        requests_counter.labels(status="validation_error").inc()
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Любая неожиданная ошибка: 500 + метрика model_error
        logger.exception("Unhandled error in /recommend")
        requests_counter.labels(status="model_error").inc()
        raise HTTPException(status_code=500, detail="internal error")
