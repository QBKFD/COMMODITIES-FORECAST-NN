"""Run every model on every commodity with several seeds and save the daily forecasts.

    python scripts/run_experiments.py                       # everything (hours on a CPU)
    python scripts/run_experiments.py --commodities gold --models transformer xgboost --seeds 0

One CSV per (commodity, model, seed) is written to
``output/predictions/<commodity>/<model>_seed<k>.csv``. Existing files are
skipped, so an interrupted run can simply be started again. Summaries and
figures are made from these files by ``scripts/summarise.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import COMMODITIES, load_commodity  # noqa: E402
from src.train_tf import walk_forward  # noqa: E402

# Model -> config file. Deterministic models need only one seed.
MODEL_CONFIGS = {
    "stacked_lstm": "con_lstm.json",
    "stacked_gru": "con_gru.json",
    "lstm_attention": "con_tft_lstm.json",
    "gru_attention": "con_tft_gru.json",
    "transformer": "con_transformers.json",
    "xgboost": "ben_xgb.json",
    "random_forest": "ben_rf.json",
    "svr": "ben_svr.json",
}
DETERMINISTIC = {"svr"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--commodities", nargs="+", default=list(COMMODITIES), choices=list(COMMODITIES))
    p.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS), choices=list(MODEL_CONFIGS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--years", nargs="+", type=int, default=[2020, 2021, 2022, 2023, 2024])
    p.add_argument("--out", type=Path, default=ROOT / "output" / "predictions")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    warnings.filterwarnings("ignore")
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")

    for commodity in args.commodities:
        df = load_commodity(commodity)
        for model in args.models:
            config = json.loads((ROOT / "config" / MODEL_CONFIGS[model]).read_text())
            seeds = args.seeds[:1] if model in DETERMINISTIC else args.seeds
            for seed in seeds:
                path = args.out / commodity / f"{model}_seed{seed}.csv"
                if path.exists():
                    logging.info("%-7s %-15s seed %d: already done", commodity, model, seed)
                    continue
                start = time.time()
                preds, _ = walk_forward(df, model, config, seed=seed, years=args.years)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                preds.to_csv(tmp, index=False)
                tmp.rename(path)  # only complete files count as done
                logging.info("%-7s %-15s seed %d: %d days in %.0fs", commodity, model, seed, len(preds),
                             time.time() - start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
