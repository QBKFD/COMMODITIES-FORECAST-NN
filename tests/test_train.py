"""End-to-end checks on a small synthetic series (fast; tiny networks, one epoch)."""

import numpy as np
import pandas as pd
import pytest

from src.data import TARGET
from src.train_tf import BENCHMARK_MODELS, NN_MODELS, make_fold, prepare_frame, walk_forward

PRICE = ["p1", "p2"]
MACRO = ["m1"]


@pytest.fixture(scope="module")
def frame():
    days = pd.bdate_range("2017-01-02", "2020-12-31")
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(len(days), 3)), index=days, columns=[*PRICE, *MACRO])
    df["p2"] += np.linspace(0, 5, len(days))  # a trend, so scaling choices matter
    df[TARGET] = rng.normal(scale=0.01, size=len(days))
    return df


CONFIG = {
    "price_features": PRICE, "macro_features": MACRO, "lookback": 5, "horizon": 1,
    "epochs": 1, "batch_size": 64, "years": [2020],
    "units_price_1": 4, "units_price_2": 4, "units_price_3": 4, "units_macro_1": 4, "units_macro_2": 4,
    "fused_dense_units_1": 4, "fused_dense_units_2": 4, "price_dense_units": 4, "macro_dense_units": 4,
    "attention_heads": 1, "attention_key_dim": 4, "macro_attention_dim": 4, "macro_attention_heads": 1,
    "n_estimators": 5,
}


def test_fold_boundaries_and_scaling(frame):
    f = prepare_frame(frame, PRICE, MACRO)
    fold = make_fold(f, PRICE, MACRO, lookback=5, year=2020)
    assert fold.test_dates.min() >= pd.Timestamp("2020-01-01")
    assert len(fold.test_dates) == len(fold.y_test_raw) > 200
    # Every training target is realised before the test year starts:
    last_train_row = f.index.searchsorted(pd.Timestamp("2020-01-01")) - 2
    n_train = len(fold.y_fit) + len(fold.y_val)
    assert n_train == last_train_row - 4 + 1
    # Validation windows come after the fitting windows.
    assert len(fold.y_val) == round(n_train * 0.15)
    # Scalers fitted on the fitting rows only: the trending column's test
    # inputs sit above the fitted range instead of being centred on it.
    assert fold.Xp_test[:, -1, 1].mean() > 1.0


@pytest.mark.parametrize("model", [*NN_MODELS, *BENCHMARK_MODELS])
def test_every_model_runs(frame, model):
    preds, _ = walk_forward(frame, model, CONFIG, seed=0)
    assert set(preds.columns) >= {"date", "year", "y_true", "y_pred", "pred_mean", "model", "seed"}
    assert len(preds) > 200 and np.isfinite(preds.y_pred).all()
    assert (preds.date.dt.year == 2020).all()


def test_old_model_names_still_work(frame):
    from src.train_tf import resolve_model_name

    assert resolve_model_name("tf_hybrid_lstm", {}) == "lstm_attention"
    assert resolve_model_name("benchmarks", {"benchmark_type": "svr"}) == "svr"
    with pytest.raises(KeyError):
        resolve_model_name("prophet", {})
