"""Is anything in the main run real once you account for how much was tested?

    python scripts/robustness.py

Uses the saved forecasts in ``output/predictions`` (seed-averaged, as in
summarise.py) and writes three tables to ``output/results``:

1. ``multiple_testing.csv``  Holm and Benjamini-Hochberg adjusted p-values
   for the 24 Diebold-Mariano tests (lower squared error than the zero
   forecast) and the 24 Pesaran-Timmermann tests (signs carry information).

2. ``model_confidence_set.csv``  Model Confidence Set (Hansen, Lunde and
   Nason 2011) per metal on squared error, over the eight models and the
   zero and training-mean forecasts. T_max statistic, moving-block bootstrap
   (block 10 days, 2000 draws). A model with MCS p-value above 0.10 is in the
   90% set, i.e. it cannot be told apart from the best one.

3. ``backtest.csv``  each model's forecast sign as a daily long/short
   position in that metal, against buy-and-hold. Gross annualised return,
   volatility and Sharpe, a Newey-West t-test that the mean daily return is
   above zero, turnover, and the break-even cost: the cost per unit traded
   (a flip from long to short trades two units) that brings the total return
   to zero. Sharpe after 1, 2 and 5 bp per unit is also reported.

The backtest assumes you can trade at the settlement price the features are
built from, and ignores roll costs, margin and financing. It is a test of
whether the sign is economically worth anything, not a trading strategy.
Figure: ``output/figures/copper_backtest.png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import bh_adjust, holm_adjust  # noqa: E402

MODELS = [
    "stacked_lstm", "stacked_gru", "lstm_attention", "gru_attention", "transformer",
    "xgboost", "random_forest", "svr",
]  # fmt: skip
COMMODITIES = ["gold", "silver", "copper"]
COSTS_BP = [1, 2, 5]
BLOCK, N_BOOT, ALPHA, SEED = 10, 2000, 0.10, 0


def load_ensembles(pred_dir: Path) -> dict[str, pd.DataFrame]:
    """Per metal: one column per model (seed-averaged forecast) plus y_true and pred_mean."""
    out = {}
    for c in COMMODITIES:
        cols = {}
        base = None
        for m in MODELS:
            files = sorted((pred_dir / c).glob(f"{m}_seed*.csv"))
            df = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in files])
            ens = df.groupby("date").agg(y_true=("y_true", "first"), pred_mean=("pred_mean", "first"),
                                         y_pred=("y_pred", "mean"))  # fmt: skip
            cols[m] = ens.y_pred
            if base is None:
                base = ens[["y_true", "pred_mean"]]
        frame = base.join(pd.DataFrame(cols), how="inner").sort_index()
        out[c] = frame
    return out


# 1. Multiple testing -----------------------------------------------------------


def multiple_testing(summary: pd.DataFrame) -> pd.DataFrame:
    s = summary[summary.model.isin(MODELS)][["commodity", "model", "rmse_vs_zero_pct",
                                              "directional_accuracy", "dm_p", "pt_p"]].copy()  # fmt: skip
    for col in ("dm_p", "pt_p"):
        s[f"{col}_holm"] = holm_adjust(s[col])
        s[f"{col}_bh"] = bh_adjust(s[col])
    return s.reset_index(drop=True)


# 2. Model Confidence Set -------------------------------------------------------


def block_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap index matrix, shape (N_BOOT, n)."""
    n_blocks = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n - BLOCK + 1, size=(N_BOOT, n_blocks))
    idx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(N_BOOT, -1)
    return idx[:, :n]


def model_confidence_set(losses: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Hansen-Lunde-Nason MCS with the T_max statistic. Returns name, MCS p-value."""
    names = list(losses.columns)
    L = losses.to_numpy()
    n = len(L)
    idx = block_indices(n, rng)
    boot_means = L[idx].mean(axis=1)  # (N_BOOT, m)
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
        p = float(np.mean(t_boot >= t.max()))
        running = max(running, p)
        worst = a[int(np.argmax(t))]
        pvals[names[worst]] = running
        alive.remove(worst)
    pvals[names[alive[0]]] = 1.0
    out = pd.DataFrame({"model": list(pvals), "mcs_p": list(pvals.values())})
    out["in_mcs_90"] = out.mcs_p > ALPHA
    out["mse_bp2"] = [float(losses[m].mean() * 1e8) for m in out.model]
    return out


def mcs_all(ens: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for c, df in ens.items():
        preds = {m: df[m] for m in MODELS}
        preds["zero"] = pd.Series(0.0, index=df.index)
        preds["mean"] = df.pred_mean
        losses = pd.DataFrame({k: (df.y_true - v) ** 2 for k, v in preds.items()})
        res = model_confidence_set(losses, rng)
        res.insert(0, "commodity", c)
        rows.append(res.sort_values("mcs_p", ascending=False))
    return pd.concat(rows, ignore_index=True)


# 3. Backtest -------------------------------------------------------------------


def nw_tstat(x: np.ndarray, lags: int = 5) -> float:
    xc = x - x.mean()
    n = len(x)
    lrv = np.sum(xc**2) / n
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * np.sum(xc[k:] * xc[:-k]) / n
    return float(x.mean() / np.sqrt(lrv / n))


def strategy_stats(ret: np.ndarray, pos: np.ndarray) -> dict:
    turnover = np.abs(np.diff(pos, prepend=0.0))
    ann = 252
    out = {
        "ann_return_pct": 100 * ret.mean() * ann,
        "ann_vol_pct": 100 * ret.std(ddof=1) * np.sqrt(ann),
        "sharpe": ret.mean() / ret.std(ddof=1) * np.sqrt(ann),
        "t_mean_gt_0": nw_tstat(ret),
        "turnover_per_year": turnover.sum() / len(ret) * ann,
        "breakeven_cost_bp": 1e4 * ret.sum() / turnover.sum() if turnover.sum() > 0 else np.nan,
    }
    out["p_mean_gt_0"] = float(1 - stats.norm.cdf(out["t_mean_gt_0"]))
    for c in COSTS_BP:
        net = ret - c * 1e-4 * turnover
        out[f"sharpe_{c}bp"] = net.mean() / net.std(ddof=1) * np.sqrt(ann)
    return out


def backtest(ens: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, curves = [], {}
    for c, df in ens.items():
        y = df.y_true.to_numpy()
        hold = strategy_stats(y, np.ones_like(y))
        rows.append({"commodity": c, "strategy": "buy_and_hold", "hit_rate": float(np.mean(y[y != 0] > 0)), **hold})
        for m in MODELS:
            pos = np.sign(df[m].to_numpy())
            ret = pos * y
            keep = y != 0
            hit = float(np.mean(np.sign(df[m].to_numpy()[keep]) == np.sign(y[keep])))
            rows.append({"commodity": c, "strategy": m, "hit_rate": hit, **strategy_stats(ret, pos)})
            if c == "copper":
                curves[m] = pd.Series(ret, index=df.index)
        if c == "copper":
            curves["buy_and_hold"] = pd.Series(y, index=df.index)
    bt = pd.DataFrame(rows)
    bt["p_mean_gt_0_bh"] = bh_adjust(np.where(bt.strategy == "buy_and_hold", np.nan, bt.p_mean_gt_0))
    return bt, pd.DataFrame(curves)


def backtest_by_year(ens: dict[str, pd.DataFrame], commodity: str, model: str) -> pd.DataFrame:
    df = ens[commodity]
    rows = []
    for year, g in df.groupby(df.index.year):
        y = g.y_true.to_numpy()
        pos = np.sign(g[model].to_numpy())
        s, h = strategy_stats(pos * y, pos), strategy_stats(y, np.ones_like(y))
        rows.append({"year": year, "sharpe": s["sharpe"], "sharpe_2bp": s["sharpe_2bp"],
                     "ann_return_pct": s["ann_return_pct"], "buy_and_hold_sharpe": h["sharpe"]})  # fmt: skip
    return pd.DataFrame(rows)


def plot_copper(curves: pd.DataFrame, ens: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    pos = np.sign(ens["stacked_gru"].to_numpy())
    turnover = np.abs(np.diff(pos, prepend=0.0))
    net2 = curves["stacked_gru"] - 2e-4 * turnover
    series = [
        (curves["stacked_gru"], "#2a78d6", "-", "Stacked GRU long/short, before costs"),
        (net2, "#2a78d6", "--", "Stacked GRU long/short, 2 bp per unit traded"),
        (curves["buy_and_hold"], "#888888", "-", "Buy and hold copper"),
    ]
    for s, color, ls, label in series:
        ax.plot(s.index, 100 * s.cumsum(), color=color, ls=ls, lw=1.6, label=label)
    ax.axhline(0, color="#bbb", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_ylabel("Cumulative log return, %")
    ax.set_title("Copper: trading the sign of the Stacked GRU forecast, 2020-2024", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, default=ROOT / "output" / "predictions")
    p.add_argument("--out", type=Path, default=ROOT / "output" / "results")
    p.add_argument("--figures", type=Path, default=ROOT / "output" / "figures")
    args = p.parse_args()

    summary = pd.read_csv(args.out / "summary.csv")
    ens = load_ensembles(args.predictions)

    mt = multiple_testing(summary)
    mt.to_csv(args.out / "multiple_testing.csv", index=False, float_format="%.4f")
    mcs = mcs_all(ens)
    mcs.to_csv(args.out / "model_confidence_set.csv", index=False, float_format="%.4f")
    bt, curves = backtest(ens)
    bt.to_csv(args.out / "backtest.csv", index=False, float_format="%.4f")
    by_year = backtest_by_year(ens, "copper", "stacked_gru")
    by_year.to_csv(args.out / "backtest_copper_gru_by_year.csv", index=False, float_format="%.4f")
    args.figures.mkdir(parents=True, exist_ok=True)
    plot_copper(curves, ens["copper"], args.figures / "copper_backtest.png")

    with pd.option_context("display.width", 240, "display.max_columns", 30, "display.max_rows", 100):
        print(mt.round(4).to_string(index=False), "\n")
        print(mcs.round(4).to_string(index=False), "\n")
        print(bt.round(3).to_string(index=False), "\n")
        print(by_year.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
