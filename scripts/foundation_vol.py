"""Zero-shot variance forecasts from pretrained Chronos-Bolt models.

    pip install chronos-forecasting           # pulls in torch
    python scripts/foundation_vol.py                          # small and base
    python scripts/foundation_vol.py --models amazon/chronos-bolt-small --device mps

Needs to download the model weights from Hugging Face. For every day from
March 2010 the model gets the last 512 squared daily returns and forecasts
the next one. The weights are never trained on these data. Writes, in the
format of run_volatility.py:

* ``<model>_median.csv``  the median forecast as returned;
* ``<model>.csv``         the median times c = mean(r² / median) over the
  training days of each test year (the QLIKE-optimal scale; see the
  amendment in docs/volatility_design.md).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_commodity  # noqa: E402
from src.volatility import FLOOR, TEST_YEARS, load, train_test_masks  # noqa: E402

CONTEXT = 512
COMMODITIES = ["gold", "silver", "copper"]


def median_forecasts(pipeline, r2_full: pd.Series, dates: pd.DatetimeIndex, batch: int) -> np.ndarray:
    import torch

    pos = r2_full.index.get_indexer(dates)
    values = r2_full.to_numpy(np.float32)
    out = np.empty(len(dates))
    for i in range(0, len(dates), batch):
        ctx = [torch.tensor(values[max(0, p - CONTEXT + 1) : p + 1]) for p in pos[i : i + batch]]
        # positional: the argument is `context` in chronos-forecasting 1.x and `inputs` in 2.x
        q, _ = pipeline.predict_quantiles(ctx, prediction_length=1, quantile_levels=[0.5])
        if not isinstance(q, torch.Tensor):
            q = torch.stack([torch.as_tensor(x) for x in q])
        out[i : i + batch] = q.reshape(len(ctx), -1)[:, 0].float().numpy()
    return np.maximum(out, FLOOR)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["amazon/chronos-bolt-small", "amazon/chronos-bolt-base"])
    p.add_argument("--commodities", nargs="+", default=COMMODITIES)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--out", type=Path, default=ROOT / "output" / "volatility" / "predictions")
    args = p.parse_args()

    import torch
    from chronos import BaseChronosPipeline

    for name in args.models:
        pipeline = BaseChronosPipeline.from_pretrained(name, device_map=args.device, torch_dtype=torch.float32)
        short = name.split("/")[-1].replace("-", "_")
        for c in args.commodities:
            t0 = time.time()
            d = load(c)
            close = load_commodity(c)["Close"]
            r2_full = (np.log(close / close.shift(1)) ** 2).dropna()
            q = median_forecasts(pipeline, r2_full, d.index, args.batch)
            y = d["y"].to_numpy()
            raw, scaled = [], []
            for year in TEST_YEARS:
                train, test = train_test_masks(d.index, year)
                scale = float(np.mean(y[train] / q[train]))
                base = {"date": d.index[test], "year": year, "y": y[test],
                        "release_day": d["release_day"].to_numpy()[test]}  # fmt: skip
                raw.append(pd.DataFrame({**base, "h": q[test]}))
                scaled.append(pd.DataFrame({**base, "h": np.maximum(scale * q[test], FLOOR)}))
                print(f"  {c} {year}: scale factor {scale:.2f}", flush=True)
            (args.out / c).mkdir(parents=True, exist_ok=True)
            cols = ["date", "year", "y", "h", "release_day"]
            pd.concat(raw)[cols].to_csv(args.out / c / f"{short}_median.csv", index=False, float_format="%.6e")
            pd.concat(scaled)[cols].to_csv(args.out / c / f"{short}.csv", index=False, float_format="%.6e")
            print(f"{time.strftime('%H:%M:%S')} {c:7s} {short}: {len(d)} forecasts in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
