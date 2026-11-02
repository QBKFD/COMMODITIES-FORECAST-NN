"""Walk-forward variance forecasts for the volatility study.

    python scripts/run_volatility.py                         # everything
    python scripts/run_volatility.py --models lstm --commodities gold --threads 1

Writes ``output/volatility/predictions/<commodity>/<model>.csv`` (networks:
``<model>_seed<k>.csv``) with columns date, year, y (squared next-day
return), h (variance forecast) and release_day. Existing files are skipped,
so the run can be resumed. Design: docs/volatility_design.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BENCHMARKS = ["hist_mean", "ewma", "har", "har_x", "garch", "gjr_garch"]
NETWORKS = ["lstm", "transformer"]
COMMODITIES = ["gold", "silver", "copper"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=BENCHMARKS + NETWORKS)
    p.add_argument("--commodities", nargs="+", default=COMMODITIES)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--threads", type=int, default=0, help="TensorFlow threads (0 = default)")
    p.add_argument("--out", type=Path, default=ROOT / "output" / "volatility" / "predictions")
    args = p.parse_args()

    if any(m in NETWORKS for m in args.models) and args.threads:
        import tensorflow as tf

        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(args.threads)

    from src.volatility import benchmark_forecasts, load, network_forecasts

    for c in args.commodities:
        d = load(c)
        (args.out / c).mkdir(parents=True, exist_ok=True)
        for m in args.models:
            jobs = [(s, args.out / c / f"{m}_seed{s}.csv") for s in args.seeds] if m in NETWORKS \
                else [(None, args.out / c / f"{m}.csv")]  # fmt: skip
            for seed, path in jobs:
                tag = f"{c:7s} {m:12s}" + (f" seed {seed}" if seed is not None else "")
                if path.exists():
                    print(f"{time.strftime('%H:%M:%S')} {tag}: already done", flush=True)
                    continue
                t0 = time.time()
                df = network_forecasts(d, m, seed) if m in NETWORKS else benchmark_forecasts(d, m)
                df.to_csv(path, index=False, float_format="%.6e")
                print(f"{time.strftime('%H:%M:%S')} {tag}: {len(df)} days in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
