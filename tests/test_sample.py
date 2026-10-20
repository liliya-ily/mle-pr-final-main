"""Тест выборки клиентов из CSV: полная история выбранных клиентов, воспроизводимость, доля."""
import pandas as pd

from recsys_utils import load_clients_sample


def make_csv(path, n_clients=2000, months=3):
    rows = [(f"2015-0{m}-28", 1_000_000 + i, i % 7) for i in range(n_clients) for m in range(1, months + 1)]
    pd.DataFrame(rows, columns=["fecha_dato", "ncodpers", "age"]).to_csv(path, index=False)


def test_sample_keeps_full_history_and_is_reproducible(tmp_path):
    path = tmp_path / "train.csv"
    make_csv(path)

    first = load_clients_sample(path, percent=20, chunksize=700)
    second = load_clients_sample(path, percent=20, chunksize=1300)

    # выбор не зависит от размера порции чтения
    assert set(first["ncodpers"]) == set(second["ncodpers"])
    # у каждого выбранного клиента сохранены все три месяца
    assert (first.groupby("ncodpers").size() == 3).all()
    # доля клиентов близка к заданной
    share = first["ncodpers"].nunique() / 2000
    assert 0.15 < share < 0.25


def test_zero_and_full_percent(tmp_path):
    path = tmp_path / "train.csv"
    make_csv(path, n_clients=100)
    assert len(load_clients_sample(path, percent=0)) == 0
    assert len(load_clients_sample(path, percent=100)) == 300
