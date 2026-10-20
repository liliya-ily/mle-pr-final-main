"""Вспомогательные функции ноутбуков: загрузка выборки клиентов, метрики топ-K, логирование запусков в MLflow."""
import mlflow
import mlflow.catboost
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature

# Метрики ранжирования, которые логируются в MLflow для каждого запуска
METRIC_KEYS = [
    "precision_at_k_all", "recall_at_k_all", "map_at_k_all", "ndcg_at_k_all",
    "precision_at_k_pos", "recall_at_k_pos", "map_at_k_pos", "ndcg_at_k_pos",
    "coverage_users", "coverage_items",
]


def load_clients_sample(csv_path, percent, id_col="ncodpers", chunksize=500_000, usecols=None):
    """Читает CSV порциями и оставляет все записи выбранной доли клиентов.

    Файл с данными весит около 2 ГБ и целиком в память не помещается без заметных затрат,
    поэтому каждая порция фильтруется сразу. Выборка детерминированная: клиент попадает в неё,
    если значение хеша от его идентификатора меньше percent (в процентах), поэтому при
    перезапуске получается тот же набор клиентов, а вся история клиента остаётся целой.
    usecols ограничивает читаемые колонки (id_col должна входить в список): так в памяти остаются
    только нужные данные.
    """
    parts = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False, usecols=usecols):
        bucket = ((chunk[id_col].astype("int64") * 2654435761) // 65536) % 100
        parts.append(chunk[bucket < percent])
    return pd.concat(parts, ignore_index=True)


def evaluate_topk(
    valid_scored,
    top_k,
    id_col,
    month_col="month",
    product_col="product_id",
    target_col="target",
):
    """Метрики топ-K рекомендаций для каждой пары (клиент, месяц).

    valid_scored содержит колонки: id_col, month_col, product_col, target_col и pred
    (оценка модели: чем больше, тем выше продукт в списке).

    Метрики считаются в двух вариантах: all (по всем парам, где есть рекомендации)
    и pos (только по парам, где у клиента появился хотя бы один новый продукт).
    MAP@K и NDCG@K учитывают позицию верного продукта в списке.
    """
    keys = [id_col, month_col]
    df_eval = valid_scored[keys + [product_col, target_col, "pred"]].copy()

    df_eval = df_eval.sort_values(keys + ["pred"], ascending=[True, True, False])
    df_eval["rank"] = df_eval.groupby(keys).cumcount() + 1

    topk_df = df_eval[df_eval["rank"] <= top_k].copy()

    gt = df_eval[df_eval[target_col] == 1].groupby(keys)[product_col].apply(set)
    pred_sets = topk_df.groupby(keys)[product_col].apply(list)

    # Precision и Recall для каждой пары (клиент, месяц)
    precisions, recalls, in_pos = [], [], []
    pos_keys = set(gt.index)
    for key, recs in pred_sets.items():
        true_set = gt.get(key, set())
        hit = len(set(recs) & true_set)
        precisions.append(hit / top_k)
        recalls.append(hit / max(1, len(true_set)))
        in_pos.append(key in pos_keys)
    precisions, recalls, in_pos = np.array(precisions), np.array(recalls), np.array(in_pos)

    # MAP@K и NDCG@K: нормировка на лучший возможный список (не длиннее K)
    rel = topk_df[target_col].astype("int32")
    topk_df["_hit_prec"] = rel * (rel.groupby([topk_df[c] for c in keys]).cumsum() / topk_df["rank"])
    topk_df["_dcg"] = rel / np.log2(topk_df["rank"] + 1)
    per_key = topk_df.groupby(keys).agg(ap_sum=("_hit_prec", "sum"), dcg=("_dcg", "sum"))
    n_true = df_eval.groupby(keys)[target_col].sum().reindex(per_key.index)
    best = np.minimum(n_true, top_k).clip(lower=1).to_numpy()
    ideal_dcg = np.cumsum(1.0 / np.log2(np.arange(2, top_k + 2)))
    has_true = (n_true > 0).to_numpy()
    ap = np.where(has_true, per_key["ap_sum"].to_numpy() / best, 0.0)
    ndcg = np.where(has_true, per_key["dcg"].to_numpy() / ideal_dcg[best - 1], 0.0)
    # порядок пар в per_key совпадает с pred_sets (обе таблицы сгруппированы по keys)
    topk_df = topk_df.drop(columns=["_hit_prec", "_dcg"])

    def mean_or_zero(values, mask=None):
        values = values if mask is None else values[mask]
        return float(values.mean()) if len(values) else 0.0

    total_pairs = df_eval[keys].drop_duplicates().shape[0]
    return {
        "precision_at_k_all": mean_or_zero(precisions),
        "recall_at_k_all": mean_or_zero(recalls),
        "map_at_k_all": mean_or_zero(ap),
        "ndcg_at_k_all": mean_or_zero(ndcg),
        "precision_at_k_pos": mean_or_zero(precisions, in_pos),
        "recall_at_k_pos": mean_or_zero(recalls, in_pos),
        "map_at_k_pos": mean_or_zero(ap, has_true),
        "ndcg_at_k_pos": mean_or_zero(ndcg, has_true),
        "coverage_users": float(len(pred_sets) / total_pairs) if total_pairs else 0.0,
        "coverage_items": (
            float(topk_df[product_col].nunique() / df_eval[product_col].nunique()) if len(df_eval) else 0.0
        ),
        "topk_df": topk_df,
        "pred_sets": pred_sets,
        "gt": gt,
    }


def print_metrics(metrics):
    """Печатает метрики, посчитанные evaluate_topk."""
    print("Precision@k (all):", round(metrics["precision_at_k_all"], 6))
    print("Recall@k    (all):", round(metrics["recall_at_k_all"], 6))
    print("MAP@k       (all):", round(metrics["map_at_k_all"], 6))
    print("NDCG@k      (all):", round(metrics["ndcg_at_k_all"], 6))
    print("Precision@k (pos):", round(metrics["precision_at_k_pos"], 6))
    print("Recall@k    (pos):", round(metrics["recall_at_k_pos"], 6))
    print("MAP@k       (pos):", round(metrics["map_at_k_pos"], 6))
    print("NDCG@k      (pos):", round(metrics["ndcg_at_k_pos"], 6))
    print("Coverage users:", round(metrics["coverage_users"], 6))
    print("Coverage items:", round(metrics["coverage_items"], 6))


def log_ranker_run(
    run_name,
    ranker,
    X_valid,
    metrics,
    params=None,
    extra_metrics=None,
    metadata=None,
    registered_model_name=None,
    n_signature_rows=5,
):
    """Логирует в MLflow параметры, метрики и модель CatBoost-ранкера.

    Если указан registered_model_name, модель регистрируется в Model Registry (новая версия).
    Возвращает (run_id, model_uri).
    """
    X_sig = X_valid.iloc[:n_signature_rows].copy()
    signature = infer_signature(X_sig, ranker.predict_proba(X_sig)[:, 1])

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params or {})
        mlflow.log_params(ranker.get_params())
        if metadata:
            mlflow.log_dict(metadata, "metadata.json")

        for key in METRIC_KEYS:
            mlflow.log_metric(key, float(metrics[key]))
        for key, value in (extra_metrics or {}).items():
            mlflow.log_metric(key, float(value))

        model_info = mlflow.catboost.log_model(
            cb_model=ranker,
            artifact_path="model",
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=X_sig,
            metadata=metadata,
        )
    return run.info.run_id, model_info.model_uri
