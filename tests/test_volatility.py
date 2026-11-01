import numpy as np
import pandas as pd
import pytest

from src.evaluation import dm_hac, model_confidence_set
from src.volatility import benchmark_forecasts, ewma, load, qlike, train_test_masks


def test_qlike_is_minimised_at_the_true_variance():
    rng = np.random.default_rng(0)
    y = (rng.standard_normal(200_000) * 0.01) ** 2  # true variance 1e-4
    grid = np.array([0.5e-4, 0.8e-4, 1e-4, 1.25e-4, 2e-4])
    losses = [qlike(y, np.full_like(y, h)).mean() for h in grid]
    assert int(np.argmin(losses)) == 2


def test_ewma_uses_only_past_returns():
    r = np.random.default_rng(1).standard_normal(300) * 0.01
    a = ewma(r)
    r2 = r.copy()
    r2[200:] *= 10  # change the future
    b = ewma(r2)
    np.testing.assert_allclose(a[:200], b[:200])


def test_train_rows_end_before_the_test_year():
    idx = pd.bdate_range("2019-12-20", "2020-01-10")
    train, test = train_test_masks(idx, 2020)
    # the last training row is the one whose next-day return is realised in 2019
    assert idx[train][-1] == pd.Timestamp("2019-12-30")
    assert idx[test][0] == pd.Timestamp("2020-01-01")


@pytest.fixture(scope="module")
def gold():
    return load("gold")


def test_target_is_the_squared_next_day_return(gold):
    np.testing.assert_allclose(gold["y"].iloc[:-1].to_numpy(), gold["ret"].iloc[1:].to_numpy() ** 2)


@pytest.mark.parametrize("model", ["har", "har_x", "garch"])
def test_benchmarks_do_not_look_ahead(gold, model):
    base = benchmark_forecasts(gold, model).set_index("date")
    cut = pd.Timestamp("2022-06-30")
    shocked = gold.copy()
    later = shocked.index > cut
    for col in ("ret", "r2_1", "y", "range2", "target"):
        shocked.loc[later, col] = shocked.loc[later, col] * 5
    other = benchmark_forecasts(shocked, model).set_index("date")
    before = base.index <= cut
    np.testing.assert_allclose(base.loc[before, "h"], other.loc[before, "h"], rtol=1e-6)


def test_dm_hac_sign():
    rng = np.random.default_rng(2)
    bench = rng.standard_normal(1000) ** 2 + 1
    better = bench - 0.2
    stat, p = dm_hac(better, bench)
    assert stat < 0 and p < 0.01


def test_mcs_keeps_the_best_model():
    rng = np.random.default_rng(3)
    n = 500
    losses = pd.DataFrame({"good": rng.normal(1.0, 0.1, n), "bad": rng.normal(2.0, 0.1, n)})
    p = model_confidence_set(losses, np.random.default_rng(0), n_boot=200)
    assert p["good"] == 1.0 and p["bad"] < 0.05
