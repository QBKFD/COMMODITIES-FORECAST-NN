"""Forecast metrics, naive baselines and significance tests.

Daily commodity returns are close to unpredictable, so a model is only
interesting if it beats the naive forecasts by more than chance:

* ``zero``  - tomorrow's return is 0 (random walk in prices)
* ``mean``  - tomorrow's return is the average return of the training period
* ``up``    - for direction only: always predict a rise

Two tests are used: Diebold-Mariano (with the Harvey-Leybourne-Newbold
small-sample correction) on squared errors against the zero forecast, and
Pesaran-Timmermann for whether the predicted signs carry information.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def directional_accuracy(y_true, y_pred) -> float:
    """Share of days where the predicted sign matches the realised one.

    Days with an exactly zero return are left out (no direction to get right).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    keep = y_true != 0
    if not keep.any():
        return float("nan")
    return float((np.sign(y_pred[keep]) == np.sign(y_true[keep])).mean())


def forecast_metrics(y_true, y_pred) -> dict[str, float]:
    """RMSE and MAE in basis points, out-of-sample R² against the zero forecast, DA."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_true - y_pred
    sse_zero = float(np.sum(y_true**2))
    return {
        "rmse_bp": float(np.sqrt(np.mean(err**2)) * 1e4),
        "mae_bp": float(np.mean(np.abs(err)) * 1e4),
        "r2_vs_zero": float(1 - np.sum(err**2) / sse_zero) if sse_zero > 0 else float("nan"),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "n": int(len(y_true)),
    }


def diebold_mariano(y_true, pred_model, pred_bench, h: int = 1) -> tuple[float, float]:
    """DM test of equal squared-error loss. Returns (statistic, one-sided p-value).

    A small p-value means the model has lower squared error than the benchmark.
    The long-run variance uses h-1 Newey-West lags; the HLN correction and a
    t distribution with n-1 degrees of freedom are used for small samples.
    """
    y_true = np.asarray(y_true, float)
    d = (y_true - np.asarray(pred_model, float)) ** 2 - (y_true - np.asarray(pred_bench, float)) ** 2
    n = len(d)
    if n < 10:
        return float("nan"), float("nan")
    d_mean = d.mean()
    dc = d - d_mean
    lrv = np.sum(dc**2) / n
    for lag in range(1, h):
        lrv += 2 * np.sum(dc[lag:] * dc[:-lag]) / n
    if lrv <= 0:
        return float("nan"), float("nan")
    dm = d_mean / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = float(dm * hln)
    return stat, float(stats.t.cdf(stat, df=n - 1))


def pesaran_timmermann(y_true, y_pred) -> tuple[float, float]:
    """PT (1992) test that predicted and realised signs are independent.

    Returns (statistic, one-sided p-value); a small p-value means the forecast
    signs carry information about the realised ones. Undefined (NaN) when the
    forecast always has the same sign, e.g. for "always up".
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    keep = y_true != 0
    a, f = (y_true[keep] > 0), (y_pred[keep] > 0)
    n = len(a)
    if n < 10:
        return float("nan"), float("nan")
    p_hat = np.mean(a == f)
    py, px = a.mean(), f.mean()
    p_star = py * px + (1 - py) * (1 - px)
    var_hat = p_star * (1 - p_star) / n
    var_star = (
        (2 * py - 1) ** 2 * px * (1 - px) / n
        + (2 * px - 1) ** 2 * py * (1 - py) / n
        + 4 * py * px * (1 - py) * (1 - px) / n**2
    )
    denom = var_hat - var_star
    if denom <= 0:
        return float("nan"), float("nan")
    stat = float((p_hat - p_star) / np.sqrt(denom))
    return stat, float(1 - stats.norm.cdf(stat))


def baseline_predictions(y_train, n_test: int) -> dict[str, np.ndarray]:
    """Naive forecasts for a test period of length ``n_test``."""
    return {
        "zero": np.zeros(n_test),
        "mean": np.full(n_test, float(np.mean(y_train))),
        "up": np.full(n_test, 1e-6),  # a tiny positive return: direction "up", error ~ zero forecast
    }
