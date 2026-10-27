"""Loading the commodity files and building model inputs without look-ahead.

Three things are handled here that the original pipeline got wrong:

1. Macro release dates. FRED dates every observation at the start of its
   period, so in the raw files the March CPI value appears on 1 March although
   it was published around 13 April. Each low-frequency series is shifted so a
   value only becomes visible once it had actually been published.
2. Market timing. The S&P 500, dollar index, VIX and the daily geopolitical
   risk index are lagged by one trading day: US equities close after the metal
   futures settle, so their same-day value is not known when the forecast for
   the next day is made.
3. Target alignment. The target on row t is the log return from the close of
   day t to the close of the next trading day, and the input window for that
   target ends on day t. (Before, the window ended one day earlier, so the
   model was effectively forecasting two days ahead.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

COMMODITIES = {
    "gold": "gold_fixed_auto.csv",
    "silver": "silver_fixed_auto.csv",
    "copper": "copper_fixed_auto.csv",
}

# Calendar days between the start of an observation period (the date FRED
# gives it) and the day the value is public. Chosen on the late side of the
# usual release calendar, so a value is never used before it existed. GDP and
# its components are also revised later; only the first release would be
# available in real time, which these files do not contain.
RELEASE_LAG_DAYS = {
    "Fed_Funds_Rate": 32,       # monthly average, published 1st business day of next month
    "1Y_Treasury_Rate": 32,     # monthly average (GS1), same
    "CPI": 45,                  # around the 10th-15th of the next month
    "Unemployment_Rate": 38,    # first Friday of the next month
    "IPI": 47,                  # mid next month
    "Consumer_Sentiment": 31,   # final reading at the end of the same month
    "M1": 56,                   # H.6 release, about four weeks after month end
    "M2": 56,
    "Housing_Starts": 48,       # around the 17th of the next month
    "Balance_Trade": 66,        # about five weeks after month end
    "GDP_Quarterly": 122,       # advance estimate, about a month after quarter end
    "Net_Exports_GS": 122,      # part of the GDP release
    "House_Price_Index": 150,   # FHFA quarterly index, about two months after quarter end
}

# Daily series from other markets: shift by this many trading days.
MARKET_LAG_ROWS = {
    col: 1
    for col in [
        "SP500_Close", "SP500_log_return", "SP500_SMA_30", "SP500_RSI_14",
        "DXY_Close", "DXY_Log_Return", "VIX_Close",
        "GPRD", "GPRD_ACT", "GPRD_THREAT", "GPRD_MA7", "GPRD_MA30",
    ]
}

TARGET = "target"


def apply_release_lag(series: pd.Series, lag_days: int) -> pd.Series:
    """Make a step-wise (monthly/quarterly) series visible only after release.

    The raw column is forward-filled from the first trading day of each
    period. Every new value is moved forward by ``lag_days`` calendar days and
    then carried forward on the trading calendar; until then the previous
    value stays in place.
    """
    s = series.dropna()
    if s.empty:
        return series
    starts = s[s.ne(s.shift())]  # first trading day of each new value
    released = pd.Series(starts.to_numpy(), index=starts.index + pd.Timedelta(days=lag_days))
    released = released[~released.index.duplicated(keep="last")]
    return released.reindex(series.index.union(released.index)).ffill().reindex(series.index)


def load_commodity(name: str, data_dir: str | Path | None = None) -> pd.DataFrame:
    """Read one commodity file and return a frame aligned for forecasting.

    Adds ``target`` = log(Close[t+1] / Close[t]). The last row has no target
    and is dropped, as are rows where a lagged macro value is not yet known.
    """
    if name not in COMMODITIES:
        raise KeyError(f"Unknown commodity {name!r}; choose from {sorted(COMMODITIES)}")
    path = Path(data_dir or DATA_DIR) / COMMODITIES[name]
    df = pd.read_csv(path, parse_dates=["ds"], thousands=",").set_index("ds").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    for col, lag in RELEASE_LAG_DAYS.items():
        if col in df:
            df[col] = apply_release_lag(df[col], lag)
    for col, rows in MARKET_LAG_ROWS.items():
        if col in df:
            df[col] = df[col].shift(rows)

    df[TARGET] = np.log(df["Close"].shift(-1) / df["Close"])
    # The original next-day dollar change; kept only for comparison.
    df = df.drop(columns=["y", "unique_id"], errors="ignore")
    return df


def make_windows(
    features: np.ndarray, target: np.ndarray, lookback: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling windows that end on the forecast day.

    Returns ``(X, y, end_index)`` where ``X[k]`` holds rows
    ``end_index[k] - lookback + 1 ... end_index[k]`` and ``y[k]`` is the target
    on row ``end_index[k]``, i.e. the return over the following day.
    """
    n = len(features)
    if n < lookback:
        return np.empty((0, lookback, features.shape[1])), np.empty(0), np.empty(0, dtype=int)
    ends = np.arange(lookback - 1, n)
    idx = ends[:, None] - np.arange(lookback - 1, -1, -1)[None, :]
    return features[idx], target[ends], ends
