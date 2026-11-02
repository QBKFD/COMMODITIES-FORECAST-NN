import numpy as np
import pytest

from src.evaluation import (
    bh_adjust,
    diebold_mariano,
    directional_accuracy,
    forecast_metrics,
    holm_adjust,
    pesaran_timmermann,
)

rng = np.random.default_rng(0)


def test_metrics_of_zero_forecast():
    y = np.array([0.01, -0.02, 0.0, 0.03])
    m = forecast_metrics(y, np.zeros(4))
    assert m["r2_vs_zero"] == 0
    assert m["rmse_bp"] == pytest.approx(np.sqrt(np.mean(y**2)) * 1e4)


def test_directional_accuracy_skips_zero_returns():
    assert directional_accuracy([0.01, -0.01, 0.0], [0.5, 0.5, -1]) == 0.5


def test_dm_detects_a_better_forecast():
    y = rng.normal(size=500)
    good = y + rng.normal(scale=0.3, size=500)
    _, p = diebold_mariano(y, good, np.zeros(500))
    assert p < 0.01
    _, p_same = diebold_mariano(y, np.zeros(500) + 1e-9, np.zeros(500))
    assert p_same > 0.05 or np.isnan(p_same)


def test_pt_detects_informative_signs_and_not_noise():
    y = rng.normal(size=1000)
    informative = np.where(rng.random(1000) < 0.7, y, -y)
    noise = rng.normal(size=1000)
    assert pesaran_timmermann(y, informative)[1] < 0.01
    assert pesaran_timmermann(y, noise)[1] > 0.01


def test_pt_undefined_for_constant_sign():
    stat, p = pesaran_timmermann(rng.normal(size=100), np.ones(100))
    assert np.isnan(stat) and np.isnan(p)


def test_bh_adjust_matches_hand_calculation():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    # sorted 0.01, 0.03, 0.04, 0.20 -> scaled 0.04, 0.06, 0.0533, 0.20 -> monotone 0.04, 0.0533, 0.0533, 0.20
    np.testing.assert_allclose(bh_adjust(p), [0.04, 0.16 / 3, 0.16 / 3, 0.20])


def test_holm_adjust_matches_hand_calculation():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    # sorted 0.01, 0.03, 0.04, 0.20 -> scaled 0.04, 0.09, 0.08, 0.20 -> monotone 0.04, 0.09, 0.09, 0.20
    np.testing.assert_allclose(holm_adjust(p), [0.04, 0.09, 0.09, 0.20])


def test_adjustments_keep_nan_and_cap_at_one():
    p = np.array([np.nan, 0.6, 0.9])
    for f in (bh_adjust, holm_adjust):
        q = f(p)
        assert np.isnan(q[0])
        assert np.all(q[1:] <= 1.0)
