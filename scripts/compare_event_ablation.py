"""Compare the forecasts with and without the economic-calendar features.

    python scripts/compare_event_ablation.py

Reads ``output/ablation/{base,events}`` (see run_event_ablation.py), averages
each model's forecasts over seeds and writes ``output/results/event_ablation.csv``:

* RMSE of both arms over 2020-2024 and the change in percent;
* one-sided Diebold-Mariano p-value for "the event features lower the
  squared error" (paired, same days);
* the same RMSE change restricted to days whose next-day return window holds
  a CPI, payroll or FOMC release, where the features should matter most;
* directional accuracy of both arms.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_commodity  # noqa: E402
from src.evaluation import diebold_mariano, directional_accuracy  # noqa: E402
from src.events import FLAG_GROUPS, add_event_features  # noqa: E402


def seed_mean(arm_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(arm_dir.glob("*/*_seed*.csv")):
        df = pd.read_csv(f, parse_dates=["date"])
        df["commodity"] = f.parent.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return (
        df.groupby(["commodity", "model", "date"])
        .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"), n_seeds=("seed", "nunique"))
        .reset_index()
    )


def rmse_bp(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)) * 1e4)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ablation", type=Path, default=ROOT / "output" / "ablation")
    p.add_argument("--out", type=Path, default=ROOT / "output" / "results" / "event_ablation.csv")
    args = p.parse_args()

    base, events = seed_mean(args.ablation / "base"), seed_mean(args.ablation / "events")
    if base.empty or events.empty:
        sys.exit(f"Need both {args.ablation}/base and {args.ablation}/events; run scripts/run_event_ablation.py")
    both = base.merge(events, on=["commodity", "model", "date"], suffixes=("_base", "_events"))
    if not np.allclose(both["y_true_base"], both["y_true_events"]):
        sys.exit("The two arms disagree on the realised returns; they were not run on the same data")

    flags = {}
    for commodity in both["commodity"].unique():
        f = add_event_features(load_commodity(commodity), commodity)
        flags[commodity] = f[[f"next_{g}" for g in FLAG_GROUPS]].max(axis=1).astype(bool)

    rows = []
    for (commodity, model), g in both.groupby(["commodity", "model"]):
        y, pb, pe = g["y_true_base"].to_numpy(), g["y_pred_base"].to_numpy(), g["y_pred_events"].to_numpy()
        ev_day = flags[commodity].reindex(g["date"]).fillna(False).to_numpy()
        r_b, r_e = rmse_bp(y, pb), rmse_bp(y, pe)
        rb_ev, re_ev = rmse_bp(y[ev_day], pb[ev_day]), rmse_bp(y[ev_day], pe[ev_day])
        rows.append(
            {
                "commodity": commodity,
                "model": model,
                "n_days": len(g),
                "seeds": int(min(g["n_seeds_base"].min(), g["n_seeds_events"].min())),
                "rmse_base_bp": round(r_b, 2),
                "rmse_events_bp": round(r_e, 2),
                "rmse_change_pct": round(100 * (r_e / r_b - 1), 2),
                "dm_p_events_better": round(diebold_mariano(y, pe, pb)[1], 3),
                "n_event_days": int(ev_day.sum()),
                "rmse_change_event_days_pct": round(100 * (re_ev / rb_ev - 1), 2),
                "da_base": round(directional_accuracy(y, pb), 3),
                "da_events": round(directional_accuracy(y, pe), 3),
            }
        )
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out.to_string(index=False))
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
