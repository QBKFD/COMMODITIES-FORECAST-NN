"""Score the volatility forecasts.

    python scripts/vol_evaluate.py

Reads ``output/volatility/predictions`` (network seeds are averaged) and
writes to ``output/volatility/results``:

* ``summary.csv``       QLIKE and MSE per metal and model, against HAR and
                        HAR-X (Diebold-Mariano, Newey-West, one-sided), Holm
                        correction across all model and metal pairs vs HAR;
* ``mcs.csv``           Model Confidence Set on QLIKE per metal (90%);
* ``by_year.csv``       QLIKE difference to HAR per test year;
* ``release_days.csv``  QLIKE difference to HAR on release days and other days;

and two figures to ``output/figures``. Design: docs/volatility_design.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cfnn.evaluation import dm_hac, holm_adjust, model_confidence_set  # noqa: E402
from cfnn.volatility import qlike  # noqa: E402

ORDER = [
    "hist_mean", "ewma", "garch", "gjr_garch", "har", "har_x", "lstm", "transformer",
    "chronos_bolt_small_median", "chronos_bolt_base_median", "chronos_bolt_small", "chronos_bolt_base",
]  # fmt: skip
LABELS = {
    "hist_mean": "Historical mean", "ewma": "EWMA (0.94)", "garch": "GARCH(1,1)",
    "gjr_garch": "GJR-GARCH", "har": "HAR", "har_x": "HAR-X", "lstm": "LSTM",
    "transformer": "Transformer", "chronos_bolt_small_median": "Chronos-Bolt small, median",
    "chronos_bolt_base_median": "Chronos-Bolt base, median", "chronos_bolt_small": "Chronos-Bolt small",
    "chronos_bolt_base": "Chronos-Bolt base",
}  # fmt: skip
LEARNED = ["lstm", "transformer", "chronos_bolt_small", "chronos_bolt_base"]
COMMODITIES = ["gold", "silver", "copper"]


def load_forecasts(pred_dir: Path, commodity: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """One column of h per model, plus y and release_day, on the days all models share."""
    frames, seeds, base = {}, {}, None
    for model in ORDER:
        files = sorted((pred_dir / commodity).glob(f"{model}.csv")) or sorted(
            (pred_dir / commodity).glob(f"{model}_seed*.csv")
        )
        if not files:
            continue
        df = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in files])
        g = df.groupby("date").agg(y=("y", "first"), release_day=("release_day", "first"), h=("h", "mean"))
        frames[model] = g["h"]
        seeds[model] = len(files)
        if base is None:
            base = g[["y", "release_day"]]
    out = base.join(pd.DataFrame(frames), how="inner").dropna()
    return out, seeds


def compare(df: pd.DataFrame, model: str, bench: str, mask=None) -> dict:
    m = np.ones(len(df), bool) if mask is None else mask
    lm, lb = qlike(df.y[m], df[model][m]), qlike(df.y[m], df[bench][m])
    stat, p = dm_hac(lm, lb)
    d = lm - lb
    n = len(d)
    dc = d - d.mean()
    lrv = np.sum(dc**2) / n + sum(2 * (1 - k / 6) * np.sum(dc[k:] * dc[:-k]) / n for k in range(1, 6))
    return {"d_qlike": float(d.mean()), "se": float(np.sqrt(lrv / n)), "p": p}


def summarise(data: dict[str, pd.DataFrame], seeds: dict) -> pd.DataFrame:
    rows = []
    for c, df in data.items():
        models = [m for m in ORDER if m in df.columns]
        mse_har = np.mean((df.y - df.har) ** 2)
        for m in models:
            r = {"commodity": c, "model": m, "n_days": len(df), "n_seeds": seeds[c][m],
                 "qlike": float(qlike(df.y, df[m]).mean()),
                 "mse_vs_har_pct": 100 * (np.mean((df.y - df[m]) ** 2) / mse_har - 1)}  # fmt: skip
            if m != "har":
                v = compare(df, m, "har")
                r.update({"d_qlike_vs_har": v["d_qlike"], "se_vs_har": v["se"], "dm_p_vs_har": v["p"]})
            if m in LEARNED:
                v = compare(df, m, "har_x")
                r.update({"d_qlike_vs_harx": v["d_qlike"], "dm_p_vs_harx": v["p"]})
            rows.append(r)
    out = pd.DataFrame(rows)
    out["dm_p_vs_har_holm"] = holm_adjust(out["dm_p_vs_har"])
    return out


def mcs(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for c, df in data.items():
        models = [m for m in ORDER if m in df.columns and not m.endswith("_median")]
        losses = pd.DataFrame({m: qlike(df.y, df[m]) for m in models})
        p = model_confidence_set(losses, rng)
        for m in models:
            rows.append({"commodity": c, "model": m, "mcs_p": p[m], "in_mcs_90": p[m] > 0.10})
    return pd.DataFrame(rows).sort_values(["commodity", "mcs_p"], ascending=[True, False])


def by_year(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for c, df in data.items():
        for year in sorted(df.index.year.unique()):
            mask = np.asarray(df.index.year == year)
            for m in [m for m in ORDER if m in df.columns and m != "har"]:
                v = compare(df, m, "har", mask)
                rows.append({"commodity": c, "year": year, "model": m, "d_qlike_vs_har": v["d_qlike"], "dm_p": v["p"]})
    return pd.DataFrame(rows)


def release_days(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for c, df in data.items():
        ev = df.release_day.to_numpy() > 0
        for m in [m for m in ORDER if m in df.columns and m != "har"]:
            a, b = compare(df, m, "har", ev), compare(df, m, "har", ~ev)
            rows.append({"commodity": c, "model": m, "n_release_days": int(ev.sum()),
                         "d_qlike_release_days": a["d_qlike"], "dm_p_release_days": a["p"],
                         "d_qlike_other_days": b["d_qlike"], "dm_p_other_days": b["p"]})  # fmt: skip
    return pd.DataFrame(rows)


def plot_dqlike(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    s = summary[(summary.model != "har") & ~summary.model.str.endswith("_median")]
    models = [m for m in ORDER if m in set(s.model)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.45 * len(models) + 1.6), sharey=True, facecolor="#fcfcfb")
    colors = {m: "#2a78d6" if m in LEARNED else "#888888" for m in models}
    for ax, c in zip(axes, COMMODITIES, strict=True):
        sub = s[s.commodity == c].set_index("model")
        for i, m in enumerate(models):
            if m not in sub.index:
                continue
            r = sub.loc[m]
            ax.errorbar(r.d_qlike_vs_har, i, xerr=1.96 * r.se_vs_har, fmt="o", color=colors[m], ms=5, lw=1.4)
        ax.axvline(0, color="#444", lw=0.8, ls="--")
        ax.set_yticks(range(len(models)), [LABELS[m] for m in models])
        ax.invert_yaxis()
        ax.set_title(c.capitalize(), loc="left", fontsize=11)
        ax.set_xlabel("QLIKE minus HAR (left of 0 beats HAR)")
        ax.set_facecolor("#fcfcfb")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Next-day variance forecasts, 2020-2024: difference to HAR with 95% intervals",
                 x=0.01, ha="left", fontsize=12)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_path(df: pd.DataFrame, path: Path, models=("har", "garch", "transformer")) -> None:
    import matplotlib.pyplot as plt

    w = df.loc["2020-01-01":"2020-06-30"]
    ann = np.sqrt(252) * 100
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.bar(w.index, np.sqrt(w.y) * ann, color="#d9d9d9", width=1.0, label="Realised |return| (annualised)")
    styles = {"har": ("#888888", "HAR"), "garch": ("#eb6834", "GARCH(1,1)"), "transformer": ("#2a78d6", "Transformer")}
    for m in models:
        if m in w.columns:
            color, label = styles[m]
            ax.plot(w.index, np.sqrt(w[m]) * ann, color=color, lw=1.6, label=label)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_ylabel("Volatility, % a year")
    ax.set_title("Gold, first half of 2020: forecast volatility for the next day", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, default=ROOT / "output" / "volatility" / "predictions")
    p.add_argument("--out", type=Path, default=ROOT / "output" / "volatility" / "results")
    p.add_argument("--figures", type=Path, default=ROOT / "output" / "figures")
    args = p.parse_args()

    data, seeds = {}, {}
    for c in COMMODITIES:
        data[c], seeds[c] = load_forecasts(args.predictions, c)
    args.out.mkdir(parents=True, exist_ok=True)
    s = summarise(data, seeds)
    s.to_csv(args.out / "summary.csv", index=False, float_format="%.5f")
    m = mcs(data)
    m.to_csv(args.out / "mcs.csv", index=False, float_format="%.4f")
    by_year(data).to_csv(args.out / "by_year.csv", index=False, float_format="%.5f")
    rd = release_days(data)
    rd.to_csv(args.out / "release_days.csv", index=False, float_format="%.5f")
    args.figures.mkdir(parents=True, exist_ok=True)
    plot_dqlike(s, args.figures / "vol_qlike_vs_har.png")
    plot_path(data["gold"], args.figures / "vol_gold_2020.png")

    cols = ["commodity", "model", "n_seeds", "qlike", "d_qlike_vs_har", "dm_p_vs_har", "dm_p_vs_har_holm",
            "d_qlike_vs_harx", "dm_p_vs_harx", "mse_vs_har_pct"]  # fmt: skip
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200):
        print(s[[c for c in cols if c in s.columns]].round(4).to_string(index=False), "\n")
        print(m.round(3).to_string(index=False), "\n")
        print(rd.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
