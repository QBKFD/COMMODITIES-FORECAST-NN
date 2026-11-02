"""Result figures. Neural networks in blue, classical benchmarks in orange,
naive baselines in grey."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

NN = ["stacked_lstm", "stacked_gru", "lstm_attention", "gru_attention", "transformer"]
BENCH = ["xgboost", "random_forest", "svr"]
LABELS = {
    "stacked_lstm": "Stacked LSTM",
    "stacked_gru": "Stacked GRU",
    "lstm_attention": "LSTM + attention",
    "gru_attention": "GRU + attention",
    "transformer": "Transformer",
    "xgboost": "XGBoost",
    "random_forest": "Random forest",
    "svr": "SVR",
    "mean": "Training mean",
}
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
COMMODITIES = ["gold", "silver", "copper"]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans", "font.size": 10,
            "axes.edgecolor": INK_3, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
            "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
            "axes.spines.top": False, "axes.spines.right": False,
            "xtick.color": INK_2, "ytick.color": INK_2, "legend.frameon": False,
        }
    )


def _color(model: str) -> str:
    return BLUE if model in NN else ORANGE if model in BENCH else INK_3


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path


def _rows(summary: pd.DataFrame) -> list[str]:
    present = set(summary["model"])
    return [m for m in [*NN, *BENCH, "mean"] if m in present]


def rmse_dotplot(summary: pd.DataFrame, preds: pd.DataFrame, path: Path) -> Path:
    """RMSE relative to the zero-change forecast, with the range over seeds."""
    _style()
    models = _rows(summary)
    commodities = [c for c in COMMODITIES if c in set(summary["commodity"])]
    fig, axes = plt.subplots(1, len(commodities), figsize=(4.2 * len(commodities), 0.42 * len(models) + 1.6),
                             sharey=True, squeeze=False)
    y = np.arange(len(models))[::-1]

    for ax, com in zip(axes[0], commodities, strict=True):
        s = summary[summary.commodity == com].set_index("model")
        zero = s.loc["zero", "rmse_bp"]
        ax.axvline(0, color=INK_3, linewidth=1, linestyle=(0, (4, 3)))
        for yi, m in zip(y, models, strict=True):
            if m not in s.index:
                continue
            v = s.loc[m, "rmse_vs_zero_pct"]
            seeds = preds[(preds.commodity == com) & (preds.model == m)]
            if seeds.seed.nunique() > 1:
                per = seeds.groupby("seed").apply(
                    lambda g, zero=zero: (np.sqrt(np.mean((g.y_true - g.y_pred) ** 2)) * 1e4 / zero - 1) * 100,
                    include_groups=False,
                )
                ax.hlines(yi, per.min(), per.max(), color=_color(m), alpha=0.35, linewidth=3)
            ax.scatter(v, yi, s=40, color=_color(m), zorder=3, edgecolor=SURFACE, linewidth=1.2)
        ax.set_title(f"{com}  (zero-change RMSE {zero:.0f} bp)", fontsize=10)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("RMSE vs. zero-change forecast (%)")
    axes[0][0].set_yticks(y, [LABELS[m] for m in models])
    axes[0][0].tick_params(axis="y", length=0)
    fig.suptitle("Next-day log-return forecasts, 2020-2024: below 0 beats the random walk",
                 x=0.0, ha="left", y=1.04, fontweight="bold", color=INK)
    fig.text(0.0, -0.04, "Dot: forecast averaged over seeds. Bar: range of single seeds. "
             "Blue: neural networks, orange: benchmarks, grey: naive baseline.", fontsize=8, color=INK_3)
    fig.tight_layout()
    return _save(fig, path)


def da_dotplot(summary: pd.DataFrame, path: Path) -> Path:
    """Directional accuracy against 50% and against always predicting a rise."""
    _style()
    models = [m for m in _rows(summary) if m != "mean"]
    commodities = [c for c in COMMODITIES if c in set(summary["commodity"])]
    fig, axes = plt.subplots(1, len(commodities), figsize=(4.2 * len(commodities), 0.42 * len(models) + 1.6),
                             sharey=True, squeeze=False)
    y = np.arange(len(models))[::-1]

    for ax, com in zip(axes[0], commodities, strict=True):
        s = summary[summary.commodity == com].set_index("model")
        up = s.loc["up", "directional_accuracy"]
        n = s.loc["zero", "n"]
        band = 1.96 * np.sqrt(0.25 / n)
        ax.axvspan(0.5 - band, 0.5 + band, color=GRID, alpha=0.7, linewidth=0)
        ax.axvline(0.5, color=INK_3, linewidth=1, linestyle=(0, (4, 3)))
        ax.axvline(up, color=INK_2, linewidth=1.2)
        ax.text(up, len(models) - 0.35, f" always up\n {up:.1%}", fontsize=7.5, color=INK_2, va="bottom")
        for yi, m in zip(y, models, strict=True):
            if m not in s.index:
                continue
            row = s.loc[m]
            ax.scatter(row.directional_accuracy, yi, s=40, color=_color(m), zorder=3,
                       edgecolor=SURFACE, linewidth=1.2)
            if row.pt_p < 0.05:
                ax.annotate("*", (row.directional_accuracy, yi), xytext=(5, -2), textcoords="offset points",
                            fontsize=11, color=INK)
        ax.set_title(com, fontsize=10)
        vals = [s.loc[m, "directional_accuracy"] for m in models if m in s.index] + [up, 0.5 - band, 0.5 + band]
        ax.set_xlim(min(vals) - 0.01, max(vals) + 0.015)
        ax.set_ylim(-0.7, len(models) + 0.6)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("directional accuracy")
    axes[0][0].set_yticks(y, [LABELS[m] for m in models])
    axes[0][0].tick_params(axis="y", length=0)
    fig.suptitle("Direction of the next-day move, 2020-2024", x=0.0, ha="left", y=1.04,
                 fontweight="bold", color=INK)
    fig.text(0.0, -0.04, "Shaded: 95% range around 50% for a coin flip. * = Pesaran-Timmermann p < 0.05.",
             fontsize=8, color=INK_3)
    fig.tight_layout()
    return _save(fig, path)
