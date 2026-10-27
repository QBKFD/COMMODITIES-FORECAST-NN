"""Turn the saved daily forecasts into result tables and figures.

    python scripts/summarise.py

Reads ``output/predictions/<commodity>/<model>_seed<k>.csv`` and writes

* ``output/results/summary.csv``  one row per commodity and model, all test
  years pooled (2020-2024), including the naive baselines and tests;
* ``output/results/by_year.csv``  the same per test year;
* figures in ``output/figures/``.

Neural network results use the average forecast over seeds (a small
ensemble); ``rmse_seed_sd`` shows how much single seeds differ.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import diebold_mariano, forecast_metrics, pesaran_timmermann  # noqa: E402

MODEL_ORDER = [
    "stacked_lstm", "stacked_gru", "lstm_attention", "gru_attention", "transformer",
    "xgboost", "random_forest", "svr",
]
LABELS = {
    "stacked_lstm": "Stacked LSTM",
    "stacked_gru": "Stacked GRU",
    "lstm_attention": "LSTM + attention",
    "gru_attention": "GRU + attention",
    "transformer": "Transformer",
    "xgboost": "XGBoost",
    "random_forest": "Random forest",
    "svr": "SVR",
    "zero": "Zero change",
    "mean": "Training mean",
    "up": "Always up",
}
COMMODITY_ORDER = ["gold", "silver", "copper"]


def load_predictions(pred_dir: Path) -> pd.DataFrame:
    files = sorted(pred_dir.glob("*/*_seed*.csv"))
    if not files:
        sys.exit(f"No prediction files in {pred_dir}; run scripts/run_experiments.py first")
    frames = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["date"])
        df["commodity"] = f.parent.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def evaluate(group: pd.DataFrame) -> dict:
    """Metrics for one commodity/model (optionally one year), seed-averaged."""
    ens = group.groupby("date").agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
    row = forecast_metrics(ens.y_true, ens.y_pred)
    row["dm_stat"], row["dm_p"] = diebold_mariano(ens.y_true, ens.y_pred, np.zeros(len(ens)))
    row["pt_stat"], row["pt_p"] = pesaran_timmermann(ens.y_true, ens.y_pred)
    per_seed = group.groupby("seed").apply(
        lambda g: np.sqrt(np.mean((g.y_true - g.y_pred) ** 2)) * 1e4, include_groups=False
    )
    row["n_seeds"] = int(per_seed.size)
    row["rmse_seed_sd"] = float(per_seed.std(ddof=0)) if per_seed.size > 1 else 0.0
    return row


def baselines(days: pd.DataFrame) -> list[dict]:
    """Zero, training-mean and always-up forecasts on the same days."""
    out = []
    for name, pred in [
        ("zero", np.zeros(len(days))),
        ("mean", days.pred_mean.to_numpy()),
        ("up", np.full(len(days), 1e-6)),
    ]:
        row = forecast_metrics(days.y_true, pred)
        if name == "zero":
            row["directional_accuracy"] = float("nan")  # predicts no direction
        if name == "mean":
            row["dm_stat"], row["dm_p"] = diebold_mariano(days.y_true, pred, np.zeros(len(days)))
        row["model"] = name
        out.append(row)
    return out


def summarise(preds: pd.DataFrame, by_year: bool) -> pd.DataFrame:
    keys = ["commodity", "model"] + (["year"] if by_year else [])
    rows = []
    for key, g in preds.groupby(keys):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True))
        row.update(evaluate(g))
        rows.append(row)
    # Baselines from the days of any one model (all models share the same days).
    group_keys = ["commodity", "year"] if by_year else ["commodity"]
    for key, g in preds.groupby(group_keys):
        days = g.drop_duplicates("date").sort_values("date")
        for b in baselines(days):
            b.update(dict(zip(group_keys, key if isinstance(key, tuple) else (key,), strict=True)))
            rows.append(b)
    out = pd.DataFrame(rows)
    out["rmse_vs_zero_pct"] = 0.0
    for key, g in out.groupby(group_keys):
        zero = g.loc[g.model == "zero", "rmse_bp"].iloc[0]
        out.loc[g.index, "rmse_vs_zero_pct"] = (g.rmse_bp / zero - 1) * 100
    order = {m: i for i, m in enumerate(MODEL_ORDER + ["zero", "mean", "up"])}
    corder = {c: i for i, c in enumerate(COMMODITY_ORDER)}
    out = out.sort_values(
        group_keys + ["model"],
        key=lambda s: s.map(corder) if s.name == "commodity" else (s.map(order) if s.name == "model" else s),
    )
    return out.reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, default=ROOT / "output" / "predictions")
    p.add_argument("--out", type=Path, default=ROOT / "output" / "results")
    p.add_argument("--figures", type=Path, default=ROOT / "output" / "figures")
    args = p.parse_args()

    preds = load_predictions(args.predictions)
    args.out.mkdir(parents=True, exist_ok=True)
    pooled = summarise(preds, by_year=False)
    yearly = summarise(preds, by_year=True)
    pooled.to_csv(args.out / "summary.csv", index=False)
    yearly.to_csv(args.out / "by_year.csv", index=False)

    from src.plotting import da_dotplot, rmse_dotplot

    rmse_dotplot(pooled, preds, args.figures / "rmse_vs_zero.png")
    da_dotplot(pooled, args.figures / "directional_accuracy.png")

    cols = ["commodity", "model", "rmse_bp", "rmse_vs_zero_pct", "r2_vs_zero", "directional_accuracy",
            "dm_p", "pt_p", "n_seeds"]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(pooled[cols].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
