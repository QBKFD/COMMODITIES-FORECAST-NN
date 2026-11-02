"""Choose hyperparameters on data before the test period.

    python scripts/tune.py run                # evaluate every candidate (resumable)
    python scripts/tune.py select             # write the winners into config/*.json

Each candidate is trained with the walk-forward scheme on the validation
years 2018 and 2019 only (training data end before each year, as in the real
evaluation), for gold, silver and copper, one seed. The score is the RMSE
divided by the RMSE of the zero forecast, averaged over years and metals;
lower is better and 1.0 means no better than "no change". The test years
2020-2024 are never used. The candidate lists are short on purpose: with a
signal this weak a large search would mostly fit noise in the validation years.

Results: ``output/tuning/<commodity>/<model>__<candidate>.json`` and, after
``select``, ``output/tuning/summary.csv``.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import COMMODITIES, load_commodity  # noqa: E402
from src.evaluation import forecast_metrics  # noqa: E402
from src.train_tf import walk_forward  # noqa: E402

VALIDATION_YEARS = [2018, 2019]
CONFIG_FILES = {
    "stacked_lstm": "con_lstm.json", "stacked_gru": "con_gru.json",
    "lstm_attention": "con_tft_lstm.json", "gru_attention": "con_tft_gru.json",
    "transformer": "con_transformers.json", "xgboost": "ben_xgb.json",
    "random_forest": "ben_rf.json", "svr": "ben_svr.json",
}  # fmt: skip

# units: width of the recurrent layers / model dimension of the Transformer.
NN_CANDIDATES = [
    {"units": 64, "dropout": 0.2, "learning_rate": 1e-3},
    {"units": 64, "dropout": 0.2, "learning_rate": 5e-4},
    {"units": 32, "dropout": 0.3, "learning_rate": 1e-3},
    {"units": 32, "dropout": 0.3, "learning_rate": 5e-4},
    {"units": 64, "dropout": 0.1, "learning_rate": 1e-3},
]
CANDIDATES = {
    **dict.fromkeys(["stacked_lstm", "stacked_gru", "lstm_attention", "gru_attention", "transformer"], NN_CANDIDATES),
    "xgboost": [{"max_depth": 2}, {"max_depth": 4}, {"max_depth": 6}],
    "random_forest": [
        {"max_depth": 5, "min_samples_leaf": 10},
        {"max_depth": 10, "min_samples_leaf": 5},
        {"max_depth": 15, "min_samples_leaf": 5},
    ],
    "svr": [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
}  # fmt: skip


def apply(model: str, config: dict, params: dict) -> dict:
    """Map a candidate onto the config keys the model reads."""
    cfg = {**config, **params}
    if model == "transformer":
        cfg["d_model"] = params["units"]
    elif model in ("lstm_attention", "gru_attention"):
        cfg["d_model"] = params["units"] // 2
    return cfg


def result_path(out: Path, commodity: str, model: str, idx: int) -> Path:
    return out / commodity / f"{model}__{idx}.json"


def run_job(commodity: str, model: str, idx: int, out: Path) -> int:
    warnings.filterwarnings("ignore")
    config = json.loads((ROOT / "config" / CONFIG_FILES[model]).read_text())
    cfg = apply(model, config, CANDIDATES[model][idx])
    start = time.time()
    preds, _ = walk_forward(load_commodity(commodity), model, cfg, seed=0, years=VALIDATION_YEARS)
    ratios = {}
    for year, g in preds.groupby("year"):
        ratios[int(year)] = forecast_metrics(g["y_true"], g["y_pred"])["rmse_bp"] / forecast_metrics(
            g["y_true"], np.zeros(len(g))
        )["rmse_bp"]
    path = result_path(out, commodity, model, idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"params": CANDIDATES[model][idx], "rmse_ratio_by_year": ratios,
                               "rmse_ratio": float(np.mean(list(ratios.values()))), "seconds": time.time() - start}))
    tmp.rename(path)
    logging.info("%-7s %-15s cand %d %s: ratio %.4f (%.0fs)", commodity, model, idx, CANDIDATES[model][idx],
                 np.mean(list(ratios.values())), time.time() - start)
    return 0


def summary(out: Path) -> pd.DataFrame:
    rows = []
    for model, cands in CANDIDATES.items():
        for idx, params in enumerate(cands):
            ratios = []
            for commodity in COMMODITIES:
                p = result_path(out, commodity, model, idx)
                if p.exists():
                    ratios.append(json.loads(p.read_text())["rmse_ratio"])
            if len(ratios) == len(COMMODITIES):
                rows.append({"model": model, "candidate": idx, **params, "score": float(np.mean(ratios))})
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["run", "select"])
    p.add_argument("--models", nargs="+", default=list(CANDIDATES), choices=list(CANDIDATES))
    p.add_argument("--commodities", nargs="+", default=list(COMMODITIES), choices=list(COMMODITIES))
    p.add_argument("--out", type=Path, default=ROOT / "output" / "tuning")
    p.add_argument("--job", nargs=3, metavar=("COMMODITY", "MODEL", "IDX"), help=argparse.SUPPRESS)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = p.parse_args()

    if args.job:
        return run_job(args.job[0], args.job[1], int(args.job[2]), args.out)

    if args.action == "run":
        failed = 0
        for model in args.models:
            for idx in range(len(CANDIDATES[model])):
                for commodity in args.commodities:
                    if result_path(args.out, commodity, model, idx).exists():
                        continue
                    cmd = [sys.executable, __file__, "run", "--out", str(args.out),
                           "--job", commodity, model, str(idx)]  # fmt: skip
                    if subprocess.run(cmd, check=False).returncode != 0:
                        logging.error("failed: %s %s %d", commodity, model, idx)
                        failed += 1
        return 1 if failed else 0

    table = summary(args.out)
    if table.empty:
        sys.exit("No complete candidates yet; run `python scripts/tune.py run` first")
    table.to_csv(args.out / "summary.csv", index=False)
    for model, g in table.groupby("model"):
        best = g.sort_values("score").iloc[0]
        path = ROOT / "config" / CONFIG_FILES[model]
        cfg = json.loads(path.read_text())
        cfg.update(CANDIDATES[model][int(best["candidate"])])
        cfg = apply(model, cfg, CANDIDATES[model][int(best["candidate"])])
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"{model:15s} -> {CANDIDATES[model][int(best['candidate'])]}  (score {best['score']:.4f}, "
              f"{len(g)} candidates, worst {g['score'].max():.4f})")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
