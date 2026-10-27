"""What the economic calendar tells us about gold, silver and copper.

    python scripts/event_analysis.py

1. Event study. For each release group, regress the return realised on the
   day the surprise became public (settlement to settlement, so the
   announcement is inside it) and the following day's return on the
   standardised surprise. Slope in basis points per one standard deviation
   of surprise, Newey-West (5 lags) t-statistics, 2010-2024. A real effect
   on day 0 and none on day 1 means the news is priced within the day.

2. Volatility. Walk-forward OLS forecasts of tomorrow's absolute return from
   a HAR-type baseline (yesterday's, last week's and last month's average
   absolute return, Corsi 2009) with and without the flags for scheduled
   CPI, payroll and FOMC releases inside tomorrow's window. Test years
   2020-2024, expanding training window starting March 2010. Paired
   Diebold-Mariano test on squared error.

Writes output/results/event_study.csv, volatility.csv and two figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import COMMODITIES, load_commodity  # noqa: E402
from src.evaluation import diebold_mariano  # noqa: E402
from src.events import FLAG_GROUPS, SURPRISE_GROUPS, add_event_features  # noqa: E402

START = "2010-03-01"
MIN_RELEASE_DAYS = 20
TEST_YEARS = [2020, 2021, 2022, 2023, 2024]
LABELS = {
    "cpi": "US CPI", "nfp": "US payrolls", "fed_rate": "Fed rate decision",
    "usd_other": "Other high-impact US", "china_pmi": "China PMI",
}  # fmt: skip


def ols_hac(y: np.ndarray, x: np.ndarray, lags: int = 5) -> tuple[float, float]:
    """Slope and Newey-West t-statistic of y on x (with intercept)."""
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ beta
    xu = X * u[:, None]
    S = xu.T @ xu
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        G = xu[k:].T @ xu[:-k]
        S += w * (G + G.T)
    bread = np.linalg.inv(X.T @ X)
    cov = bread @ S @ bread
    return float(beta[1]), float(beta[1] / np.sqrt(cov[1, 1]))


def event_study(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for commodity, df in frames.items():
        d = df.loc[START:"2024-11-25"]
        same_day, next_day = d["log_return"].to_numpy(), d["target"].to_numpy()
        for group in SURPRISE_GROUPS:
            z = d[f"z_{group}"].to_numpy()
            if (z != 0).sum() < MIN_RELEASE_DAYS:  # e.g. Fed decisions: too few scored releases
                continue
            mask = np.isfinite(same_day) & np.isfinite(next_day)
            b0, t0 = ols_hac(same_day[mask] * 1e4, z[mask])
            b1, t1 = ols_hac(next_day[mask] * 1e4, z[mask])
            rows.append(
                {
                    "commodity": commodity, "group": group, "days_with_release": int((z != 0).sum()),
                    "same_day_bp": b0, "same_day_t": t0, "next_day_bp": b1, "next_day_t": t1,
                }
            )  # fmt: skip
    return pd.DataFrame(rows)


def volatility(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for commodity, df in frames.items():
        d = df.loc[START:].copy()
        a = d["log_return"].abs()
        d["abs_1"], d["abs_5"], d["abs_22"] = a, a.rolling(5).mean(), a.rolling(22).mean()
        d["y"] = d["target"].abs()
        flags = [f"next_{g}" for g in FLAG_GROUPS]
        d["any_event"] = d[flags].max(axis=1)
        d = d.dropna(subset=["abs_1", "abs_5", "abs_22", "y"])
        har = ["abs_1", "abs_5", "abs_22"]
        specs = {"mean": [], "har": har, "har_events": har + flags}
        preds = {k: [] for k in specs}
        truth, year_of, event_day = [], [], []
        for year in TEST_YEARS:
            train = d.loc[: f"{year - 1}-12-29"]  # last target realised before the test year
            test = d.loc[str(year)]
            truth.append(test["y"].to_numpy())
            year_of.append(np.full(len(test), year))
            event_day.append(test["any_event"].to_numpy() > 0)
            for name, cols in specs.items():
                X = np.column_stack([np.ones(len(train))] + [train[c] for c in cols])
                beta = np.linalg.lstsq(X, train["y"].to_numpy(), rcond=None)[0]
                Xt = np.column_stack([np.ones(len(test))] + [test[c] for c in cols])
                preds[name].append(np.maximum(Xt @ beta, 1e-5))
        y = np.concatenate(truth)
        ev = np.concatenate(event_day)
        p = {k: np.concatenate(v) for k, v in preds.items()}
        rmse = {k: np.sqrt(np.mean((y - v) ** 2)) * 1e4 for k, v in p.items()}
        dm_har = diebold_mariano(y, p["har"], p["mean"])[1]
        dm_ev = diebold_mariano(y, p["har_events"], p["har"])[1]
        sse = lambda k, m: np.sum((y[m] - p[k][m]) ** 2)  # noqa: E731
        rows.append(
            {
                "commodity": commodity, "n_days": len(y), "n_event_days": int(ev.sum()),
                "rmse_mean_bp": rmse["mean"], "rmse_har_bp": rmse["har"], "rmse_har_events_bp": rmse["har_events"],
                "har_vs_mean_pct": 100 * (rmse["har"] / rmse["mean"] - 1), "dm_p_har_better": dm_har,
                "events_vs_har_pct": 100 * (rmse["har_events"] / rmse["har"] - 1), "dm_p_events_better": dm_ev,
                "dm_p_events_better_event_days": diebold_mariano(y[ev], p["har_events"][ev], p["har"][ev])[1],
                "events_vs_har_event_days_pct": 100 * (np.sqrt(sse("har_events", ev) / sse("har", ev)) - 1),
                "events_vs_har_other_days_pct": 100 * (np.sqrt(sse("har_events", ~ev) / sse("har", ~ev)) - 1),
                "mean_abs_return_event_days_bp": float(y[ev].mean() * 1e4),
                "mean_abs_return_other_days_bp": float(y[~ev].mean() * 1e4),
            }
        )  # fmt: skip
    return pd.DataFrame(rows)


def plot_event_study(es: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharex=True, facecolor="#fcfcfb")
    groups = [g for g in SURPRISE_GROUPS if g in set(es.group)]
    for ax, commodity in zip(axes, COMMODITIES, strict=True):
        sub = es[(es.commodity == commodity) & es.group.isin(groups)].set_index("group").loc[groups]
        y = np.arange(len(groups))
        for off, col, color, label in [(-0.18, "same_day", "#2a78d6", "Return on the release day"),
                                       (0.18, "next_day", "#eb6834", "Return on the next day")]:  # fmt: skip
            b, t = sub[f"{col}_bp"].to_numpy(), sub[f"{col}_t"].to_numpy()
            ax.errorbar(b, y + off, xerr=1.96 * np.abs(b / t), fmt="o", color=color, ms=5, lw=1.5, label=label)
        ax.axvline(0, color="#888", lw=0.8)
        ax.set_yticks(y, [LABELS[g] for g in groups] if ax is axes[0] else [])
        ax.set_title(commodity.capitalize(), loc="left", fontsize=11)
        ax.invert_yaxis()
        ax.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("bp per 1 sd of surprise")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=2)
    fig.suptitle("Price reaction to release surprises, same day and next day (95% intervals)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    frames = {c: add_event_features(load_commodity(c), c) for c in COMMODITIES}
    out = ROOT / "output" / "results"
    fig_dir = ROOT / "output" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    es = event_study(frames)
    es.to_csv(out / "event_study.csv", index=False, float_format="%.3f")
    vol = volatility(frames)
    vol.to_csv(out / "volatility.csv", index=False, float_format="%.3f")
    plot_event_study(es, fig_dir / "event_study.png")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(es.round(2).to_string(index=False))
        print()
        print(vol.round(2).T.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
