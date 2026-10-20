"""Тест метрик топ-K из notebooks/recsys_utils.py на примере с ручным расчётом."""
import numpy as np
import pandas as pd
import pytest

from recsys_utils import evaluate_topk


@pytest.fixture()
def scored():
    """Два клиента в одном месяце, пять кандидатов у каждого.

    Клиент 1: верные продукты p1 (позиция 1) и p3 (позиция 3) при K = 3.
    Клиент 2: верных продуктов нет.
    """
    rows = []
    products = ["p1", "p2", "p3", "p4", "p5"]
    preds = [0.9, 0.8, 0.7, 0.6, 0.5]
    month = pd.Timestamp("2016-05-01")
    for user, targets in ((1, [1, 0, 1, 0, 0]), (2, [0, 0, 0, 0, 0])):
        for product, pred, target in zip(products, preds, targets):
            rows.append((user, month, product, target, pred))
    return pd.DataFrame(rows, columns=["ncodpers", "month", "product_id", "target", "pred"])


def test_topk_metrics_match_manual_calculation(scored):
    metrics = evaluate_topk(scored, top_k=3, id_col="ncodpers")

    ap = (1 / 1 + 2 / 3) / 2                                    # AP@3 клиента 1
    dcg = 1 / np.log2(2) + 1 / np.log2(4)
    ndcg = dcg / (1 / np.log2(2) + 1 / np.log2(3))              # NDCG@3 клиента 1

    # по клиентам, у которых есть новые продукты (pos)
    assert metrics["precision_at_k_pos"] == pytest.approx(2 / 3)
    assert metrics["recall_at_k_pos"] == pytest.approx(1.0)
    assert metrics["map_at_k_pos"] == pytest.approx(ap)
    assert metrics["ndcg_at_k_pos"] == pytest.approx(ndcg)

    # по всем клиентам (all): второй клиент даёт нули
    assert metrics["precision_at_k_all"] == pytest.approx(1 / 3)
    assert metrics["recall_at_k_all"] == pytest.approx(0.5)
    assert metrics["map_at_k_all"] == pytest.approx(ap / 2)
    assert metrics["ndcg_at_k_all"] == pytest.approx(ndcg / 2)

    assert metrics["coverage_users"] == pytest.approx(1.0)
    assert metrics["coverage_items"] == pytest.approx(3 / 5)


def test_perfect_ranking_gives_one(scored):
    scored.loc[scored["ncodpers"] == 1, "pred"] = [0.9, 0.5, 0.8, 0.4, 0.3]  # верные продукты на позициях 1 и 2
    metrics = evaluate_topk(scored, top_k=2, id_col="ncodpers")
    assert metrics["map_at_k_pos"] == pytest.approx(1.0)
    assert metrics["ndcg_at_k_pos"] == pytest.approx(1.0)
    assert metrics["precision_at_k_pos"] == pytest.approx(1.0)
