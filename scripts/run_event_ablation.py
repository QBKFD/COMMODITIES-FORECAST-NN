"""Train every model twice, with and without the economic-calendar features.

    python scripts/build_event_features.py --raw data/calendar_raw   # once
    python scripts/run_event_ablation.py                              # both arms
    python scripts/run_event_ablation.py --models xgboost --seeds 0   # quick check
    python scripts/compare_event_ablation.py                          # tables

Both arms start training in March 2010, the first date with a full calendar
history behind it, so they differ only in the event columns. The forecasts go to
``output/ablation/{base,events}/<commodity>/<model>_seed<k>.csv``; the run
is resumable like ``run_experiments.py``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.events import EVENTS_FILE  # noqa: E402

DEFAULT_MODELS = ["gru_attention", "transformer", "xgboost", "random_forest"]
TRAIN_START = "2010-03-01"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--commodities", nargs="+", default=["gold", "silver", "copper"])
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    p.add_argument("--arms", nargs="+", default=["base", "events"], choices=["base", "events"])
    p.add_argument("--out", type=Path, default=ROOT / "output" / "ablation")
    args = p.parse_args()

    if "events" in args.arms and not EVENTS_FILE.exists():
        sys.exit(f"{EVENTS_FILE} not found; run scripts/build_event_features.py first")

    status = 0
    # Alternate the arms per model so a partial run still gives paired results.
    for commodity in args.commodities:
        for model in args.models:
            for arm in args.arms:
                cmd = [
                    sys.executable, str(ROOT / "scripts" / "run_experiments.py"),
                    "--commodities", commodity, "--models", model, "--seeds", *args.seeds,
                    "--train-start", TRAIN_START, "--out", str(args.out / arm),
                ]
                if arm == "events":
                    cmd.append("--events")
                status |= subprocess.run(cmd).returncode
    return status


if __name__ == "__main__":
    sys.exit(main())
