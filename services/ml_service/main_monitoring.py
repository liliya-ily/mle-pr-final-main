# services/ml_service/main_monitoring.py

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI, Body, HTTPException

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter

from .fast_api_handler import FastApiHandler

logger = logging.getLogger("uvicorn.error")


def _artifacts_dir() -> Path:
    """
    Папка с артефактами сервиса: models/model.cbm и assets/ (parquet и json).
    По умолчанию это папка services рядом с кодом сервиса; в контейнере в неё подключаются
    тома models и assets (см. docker-compose.yaml). Можно переопределить переменной ARTIFACTS_DIR.
    """
    default = Path(__file__).resolve().parents[1]
    return Path(os.getenv("ARTIFACTS_DIR", default))


def _require(path: Path) -> Path:
    """
    Возвращает путь к артефакту или понятную ошибку, если файла нет.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Не найден артефакт {path}. Создайте его, выполнив ноутбук notebooks/02_modeling.ipynb"
        )
    return path


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan FastAPI:
    - при старте: загрузка артефактов с диска в память и создание handler
    - при остановке: ничего специального не требуется
    """
    base = _artifacts_dir()
    logger.info("Loading artifacts from: %s", base)

    # config
    cfg_path = _require(base / "assets" / "inference_config.json")
    infer_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # data
    personal_als = pd.read_parquet(_require(base / "assets" / "personal_als.parquet"))
    user_features = pd.read_parquet(_require(base / "assets" / "user_features.parquet"))

    # нормализация типов для join/фильтров
    personal_als["month_str"] = personal_als["month_str"].astype(str)
    user_features["month_str"] = user_features["month_str"].astype(str)

    # model
    model = CatBoostClassifier()
    model.load_model(str(_require(base / "models" / "model.cbm")))

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
def recommend(payload: dict = Body(..., example={"user_id": 15889, "month": "2016-01", "k": 5})):
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
