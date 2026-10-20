"""Общие фикстуры тестов: маленькие артефакты сервиса (модель CatBoost и таблицы) во временной папке."""
import json

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier

ID_COL = "ncodpers"
PRODUCTS = ["ind_cco_fin_ult1", "ind_recibo_ult1", "ind_tjcr_fin_ult1", "ind_nomina_ult1"]
MONTH = "2016-01"
FEATURES = ["score_als", "month_str", "product_id", "age"]
CAT_FEATURES = ["month_str", "product_id"]


@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory):
    """Создаёт services/models/model.cbm и services/assets/* в стиле реальных артефактов."""
    base = tmp_path_factory.mktemp("artifacts")
    (base / "models").mkdir()
    (base / "assets").mkdir()
    rng = np.random.default_rng(0)

    # кандидаты ALS: 3 клиента, у каждого все продукты с убывающим score
    rows = []
    for user in (101, 102, 103):
        for rank, product in enumerate(PRODUCTS, start=1):
            rows.append((user, MONTH, product, float(len(PRODUCTS) - rank + rng.random() * 0.1), rank))
    personal_als = pd.DataFrame(rows, columns=[ID_COL, "month_str", "product_id", "score_als", "rank_als"])
    personal_als["score_als"] = personal_als["score_als"].astype("float32")
    personal_als.to_parquet(base / "assets" / "personal_als.parquet", index=False)

    user_features = pd.DataFrame({ID_COL: [101, 102, 103], "month_str": [MONTH] * 3, "age": [30.0, 45.0, 52.0]})
    user_features.to_parquet(base / "assets" / "user_features.parquet", index=False)

    config = {"top_k": 3, "features": FEATURES, "cat_features": CAT_FEATURES}
    (base / "assets" / "inference_config.json").write_text(json.dumps(config), encoding="utf-8")

    # маленькая модель на тех же признаках: чем выше score_als, тем вероятнее покупка
    train = personal_als.merge(user_features.drop(columns=["month_str"]), on=ID_COL)
    X = train[FEATURES].copy()
    for col in CAT_FEATURES:
        X[col] = X[col].astype("string")
    y = (train["rank_als"] <= 2).astype(int)
    model = CatBoostClassifier(iterations=20, depth=3, verbose=0, random_seed=42, allow_writing_files=False)
    model.fit(X, y, cat_features=CAT_FEATURES)
    model.save_model(str(base / "models" / "model.cbm"))
    return base
