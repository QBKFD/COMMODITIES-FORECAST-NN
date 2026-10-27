"""Run every model on every commodity with several seeds and save the daily forecasts.

    python scripts/run_experiments.py                       # everything (hours on a CPU)
    python scripts/run_experiments.py --commodities gold --models transformer xgboost --seeds 0

Options for the event-calendar ablation (see scripts/run_event_ablation.py):
``--train-start`` drops data before a date, ``--events`` adds the release
surprises and scheduled-event flags from ``src/events.py`` to the macro inputs.

One CSV per (commodity, model, seed) is written to
``output/predictions/<commodity>/<model>_seed<k>.csv``. Existing files are
skipped, so an interrupted run can simply be started again. Every job runs in
its own Python process: TensorFlow does not give memory back between models,
and a single long process grew past 3 GB and was killed. Summaries and
figures are made from these files by ``scripts/summarise.py``.
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


def job_path(out: Path, commodity: str, model: str, seed: int) -> Path:
    return out / commodity / f"{model}_seed{seed}.csv"


def run_job(
    commodity: str,
    model: str,
    seed: int,
    years: list[int],
    out: Path,
    train_start: str | None = None,
    events: bool = False,
) -> int:
    warnings.filterwarnings("ignore")
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    config = json.loads((ROOT / "config" / MODEL_CONFIGS[model]).read_text())
    df = load_commodity(commodity)
    if train_start:
        config["train_start"] = train_start
    if events:
        from src.events import EVENT_COLUMNS, add_event_features

        df = add_event_features(df, commodity)
        config["macro_features"] = [*config["macro_features"], *EVENT_COLUMNS]
    start = time.time()
    preds, _ = walk_forward(df, model, config, seed=seed, years=years)
    path = job_path(out, commodity, model, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    preds.to_csv(tmp, index=False)
    tmp.rename(path)  # only complete files count as done
    logging.info("%-7s %-15s seed %d: %d days in %.0fs", commodity, model, seed, len(preds), time.time() - start)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--commodities", nargs="+", default=list(COMMODITIES), choices=list(COMMODITIES))
    p.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS), choices=list(MODEL_CONFIGS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--years", nargs="+", type=int, default=[2020, 2021, 2022, 2023, 2024])
    p.add_argument("--out", type=Path, default=ROOT / "output" / "predictions")
    p.add_argument("--train-start", help="drop data before this date, e.g. 2010-03-01")
    p.add_argument("--events", action="store_true", help="add the economic-calendar features")
    p.add_argument("--in-process", action="store_true", help=argparse.SUPPRESS)  # used for the child jobs
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = p.parse_args()

    if args.in_process:
        return run_job(
            args.commodities[0], args.models[0], args.seeds[0], args.years, args.out, args.train_start, args.events
        )

    failed = []
    for commodity in args.commodities:
        for model in args.models:
            seeds = args.seeds[:1] if model in DETERMINISTIC else args.seeds
            for seed in seeds:
                if job_path(args.out, commodity, model, seed).exists():
                    logging.info("%-7s %-15s seed %d: already done", commodity, model, seed)
                    continue
                cmd = [sys.executable, __file__, "--in-process", "--commodities", commodity, "--models", model,
                       "--seeds", str(seed), "--out", str(args.out), "--years", *map(str, args.years)]
                if args.train_start:
                    cmd += ["--train-start", args.train_start]
                if args.events:
                    cmd.append("--events")
                if subprocess.run(cmd).returncode != 0:
                    logging.error("%-7s %-15s seed %d: failed", commodity, model, seed)
                    failed.append((commodity, model, seed))
    if failed:
        logging.error("%d job(s) failed: %s", len(failed), failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
