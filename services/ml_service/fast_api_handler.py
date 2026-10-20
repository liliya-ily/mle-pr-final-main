# services/ml_service/fast_api_handler.py

import pandas as pd


class FastApiHandler:
    """
    Обработчик бизнес-логики для /recommend.

    Все зависимости передаются готовыми объектами через __init__:
    - model: обученная CatBoost модель
    - personal_als: кандидаты ALS (top-K) по user-month
    - user_features: признаки пользователя по user-month
    - infer_cfg: конфиг инференса (списки признаков и т.п.)
    """

    def __init__(
        self,
        model,
        personal_als: pd.DataFrame,
        user_features: pd.DataFrame,
        infer_cfg: dict,
        id_col: str = "ncodpers",
        month_col: str = "month_str",
    ):
        self.model = model
        self.personal_als = personal_als
        self.user_features = user_features
        self.infer_cfg = infer_cfg
        self.id_col = id_col
        self.month_col = month_col

    def _validate_payload(self, payload: dict) -> tuple[int, str, int]:
        """
        Валидация входных параметров. Возвращает нормализованные (user_id, month, k).

        Требования:
        - user_id: int
        - month: str "YYYY-MM"
        - k: int > 0
        """
        if not isinstance(payload, dict):
            raise ValueError("payload must be dict")

        if "user_id" not in payload:
            raise ValueError("user_id is required")
        if "month" not in payload:
            raise ValueError("month is required")

        user_id = payload["user_id"]
        month = payload["month"]
        k = payload.get("k", self.infer_cfg.get("top_k", 5))

        if not isinstance(user_id, int):
            raise ValueError("user_id must be int")
        if not isinstance(month, str) or len(month) != 7 or month[4] != "-":
            raise ValueError("month must be string in format YYYY-MM")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be positive int")

        return user_id, month, k

    def _build_feature_table(self, user_id: int, month: str) -> pd.DataFrame:
        """
        Собирает таблицу признаков для ранжирования:
        - берутся кандидаты из personal_als для (user_id, month)
        - джойнятся признаки пользователя из user_features для (user_id, month)
        - формируется X с колонками из infer_cfg["features"] (если список есть)
        """
        pa = self.personal_als
        uf = self.user_features

        df_u = pa[(pa[self.id_col] == user_id) & (pa[self.month_col] == month)].copy()
        if df_u.empty:
            return df_u

        uf_row = uf[(uf[self.id_col] == user_id) & (uf[self.month_col] == month)]
        if uf_row.empty:
            return pd.DataFrame()

        df = df_u.merge(
            uf_row.drop(columns=[self.month_col], errors="ignore"),
            on=self.id_col,
            how="left",
        )

        features = self.infer_cfg.get("features")
        if isinstance(features, list) and len(features) > 0:
            missing = [c for c in features if c not in df.columns]
            if missing:
                raise ValueError(f"missing required feature columns: {missing}")
            df = df[features].copy()

        return df

    def handle(self, payload: dict) -> dict:
        """
        Основная точка входа для API.
        Возвращает структуру:
        {"user_id": ..., "month": "...", "recommendations": [{"product_id": ..., "score": ..., "score_als": ...}, ...]}
        """
        user_id, month, k = self._validate_payload(payload)

        X = self._build_feature_table(user_id=user_id, month=month)
        if X.empty:
            return {"user_id": user_id, "month": month, "recommendations": []}

        cat_features = self.infer_cfg.get("cat_features", [])
        if isinstance(cat_features, list) and len(cat_features) > 0:
            for c in cat_features:
                if c in X.columns:
                    X[c] = X[c].astype("string")

        proba = self.model.predict_proba(X)[:, 1]

        out = X.copy()
        out["_score"] = proba

        # Сохранение score_als в ответе полезно для дебага.
        # Важный момент: score_als должен существовать в X, если он был среди features.
        keep_cols = []
        if "product_id" in out.columns:
            keep_cols.append("product_id")
        if "score_als" in out.columns:
            keep_cols.append("score_als")
        keep_cols.append("_score")

        out = out[keep_cols].sort_values("_score", ascending=False).head(k)

        recs = []
        for _, r in out.iterrows():
            rec = {"product_id": str(r["product_id"]), "score": float(r["_score"])}
            if "score_als" in out.columns:
                rec["score_als"] = float(r["score_als"])
            recs.append(rec)

        return {"user_id": user_id, "month": month, "recommendations": recs}
