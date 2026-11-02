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


def bh_adjust(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values); NaNs are left as NaN."""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    scaled = pv[order] * m / np.arange(1, m + 1)
    adj = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
    out = np.empty(m)
    out[order] = adj
    q[ok] = out
    return q


def holm_adjust(p) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (family-wise error); NaNs are left as NaN."""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    scaled = pv[order] * (m - np.arange(m))
    adj = np.minimum(np.maximum.accumulate(scaled), 1.0)
    out = np.empty(m)
    out[order] = adj
    q[ok] = out
    return q


def dm_hac(loss_model, loss_bench, lags: int = 5) -> tuple[float, float]:
    """DM test on any per-day loss with a Newey-West long-run variance.

    Returns (statistic, one-sided p-value); a small p-value means the model
    has the lower average loss. Used for volatility forecasts, whose loss
    differences are autocorrelated.
    """
    d = np.asarray(loss_model, float) - np.asarray(loss_bench, float)
    n = len(d)
    if n < 10:
        return float("nan"), float("nan")
    dc = d - d.mean()
    lrv = np.sum(dc**2) / n
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * np.sum(dc[k:] * dc[:-k]) / n
    if lrv <= 0:
        return float("nan"), float("nan")
    stat = float(d.mean() / np.sqrt(lrv / n))
    return stat, float(stats.norm.cdf(stat))


def block_bootstrap_indices(n: int, n_boot: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap index matrix, shape (n_boot, n)."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)
    return idx[:, :n]


def model_confidence_set(
    losses, rng: np.random.Generator, n_boot: int = 2000, block: int = 10
) -> dict[str, float]:
    """Model Confidence Set p-values (Hansen, Lunde and Nason 2011), T_max statistic.

    ``losses`` is a DataFrame with one column of per-day losses per model.
    Models are eliminated one at a time, worst first; a model's MCS p-value is
    the largest elimination p-value up to its own. Models with a p-value above
    alpha form the (1 - alpha) confidence set.
    """
    names = list(losses.columns)
    L = np.asarray(losses, float)
    idx = block_bootstrap_indices(len(L), n_boot, block, rng)
    boot_means = L[idx].mean(axis=1)
    means = L.mean(axis=0)
    alive = list(range(len(names)))
    pvals, running = {}, 0.0
    while len(alive) > 1:
        a = np.array(alive)
        d = means[a] - means[a].mean()
        d_boot = boot_means[:, a] - boot_means[:, a].mean(axis=1, keepdims=True)
        se = np.sqrt(np.mean((d_boot - d) ** 2, axis=0))
        t = d / se
        t_boot = ((d_boot - d) / se).max(axis=1)
        running = max(running, float(np.mean(t_boot >= t.max())))
        worst = a[int(np.argmax(t))]
        pvals[names[worst]] = running
        alive.remove(worst)
    pvals[names[alive[0]]] = 1.0
    return pvals


def baseline_predictions(y_train, n_test: int) -> dict[str, np.ndarray]:
    """Naive forecasts for a test period of length ``n_test``."""
    return {
        "zero": np.zeros(n_test),
        "mean": np.full(n_test, float(np.mean(y_train))),
        "up": np.full(n_test, 1e-6),  # a tiny positive return: direction "up", error ~ zero forecast
    }
