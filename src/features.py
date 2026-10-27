"""Stationary model inputs built from the raw columns.

The raw files contain price levels (Open, High, Low, Close, moving averages,
ATR, MACD in dollars) and macro levels (CPI index, M2 in billions). Their
scale drifts over the sample: gold is near 400 in 2003 and 2,700 in 2024, so
in the later test years the scaled inputs fall far outside anything seen in
training. The features below carry the same information in a form that means
the same thing in every year: returns, ratios to the current price, distances
from moving averages, and year-on-year changes.

Everything uses data up to and including the forecast day only. The release
lags of ``src/data.py`` are applied first, so macro columns here are the
already lagged ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

PRICE_FEATURES = [
    "log_return", "log_return_lag1", "log_return_lag2",
    "open_gap", "hl_range", "close_in_range", "ret_5", "ret_20",
    "dist_sma7", "dist_sma30", "macd_pct", "atr_pct", "bb_width_pct",
    "RSI_14", "pos_in_range", "trend_up", "strong_momentum",
    "rolling_vol_5", "rolling_vol_20", "abs_ret", "Volume_Zscore",
]  # fmt: skip

MACRO_FEATURES = [
    # one-year changes (the levels of rates and unemployment differ a lot between regimes)
    "Fed_Funds_chg", "T1Y_chg", "Unemployment_chg", "Consumer_Sentiment",
    # year-on-year changes of trending series (M1 is left out: it was redefined in May 2020)
    "CPI_yoy", "M2_yoy", "IPI_yoy", "GDP_yoy", "Balance_Trade_yoy",
    # markets (already lagged one trading day)
    "SP500_log_return", "SP500_dist_sma30", "SP500_RSI_14",
    "DXY_Log_Return", "DXY_dist_60", "VIX_Close", "VIX_change",
    "GPRD_log", "GPRD_ACT_log", "GPRD_THREAT_log", "GPRD_MA30_log",
]  # fmt: skip


def _yoy(series: pd.Series) -> pd.Series:
    """Log change over one year of trading days; series with sign changes use the plain difference."""
    s = series.astype(float)
    if (s > 0).all():
        return np.log(s / s.shift(TRADING_DAYS_PER_YEAR))
    return (s - s.shift(TRADING_DAYS_PER_YEAR)) / s.abs().rolling(TRADING_DAYS_PER_YEAR, min_periods=20).mean()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with the stationary features added (original columns are kept)."""
    out = df.copy()
    close = out["Close"].astype(float)

    out["open_gap"] = np.log(out["Open"] / close.shift(1))
    out["hl_range"] = np.log(out["High"] / out["Low"].where(out["Low"] > 0))
    spread = (out["High"] - out["Low"]).replace(0, np.nan)
    # a few rows in the copper file have a Close outside High/Low; clip to the day's range
    out["close_in_range"] = ((close - out["Low"]) / spread).clip(0, 1).fillna(0.5)
    out["abs_ret"] = out["log_return"].abs()
    out["ret_5"] = np.log(close / close.shift(5))
    out["ret_20"] = np.log(close / close.shift(20))
    out["dist_sma7"] = np.log(close / out["SMA_7"])
    out["dist_sma30"] = np.log(close / out["SMA_30"])
    out["macd_pct"] = out["MACD"] / close
    out["atr_pct"] = out["ATR_14"] / close
    # Bollinger width in the raw file is in price units (and in some years a
    # different scale); recompute it relative to the 20-day mean.
    out["bb_width_pct"] = 4 * close.rolling(20).std() / close.rolling(20).mean()

    for col, name in [
        ("CPI", "CPI_yoy"), ("M2", "M2_yoy"), ("IPI", "IPI_yoy"),
        ("GDP_Quarterly", "GDP_yoy"), ("Balance_Trade", "Balance_Trade_yoy"),
    ]:  # fmt: skip
        out[name] = _yoy(out[col])
    for col, name in [("Fed_Funds_Rate", "Fed_Funds_chg"), ("1Y_Treasury_Rate", "T1Y_chg"),
                      ("Unemployment_Rate", "Unemployment_chg")]:  # fmt: skip
        out[name] = out[col] - out[col].shift(TRADING_DAYS_PER_YEAR)

    out["SP500_dist_sma30"] = np.log(out["SP500_Close"] / out["SP500_SMA_30"])
    out["DXY_dist_60"] = np.log(out["DXY_Close"] / out["DXY_Close"].rolling(60).mean())
    out["VIX_change"] = np.log(out["VIX_Close"] / out["VIX_Close"].shift(1))
    for col in ("GPRD", "GPRD_ACT", "GPRD_THREAT", "GPRD_MA30"):
        out[f"{col}_log"] = np.log1p(out[col])

    # replace infinities from zero prices/volumes by missing values
    new = [c for c in out.columns if c not in df.columns]
    out[new] = out[new].replace([np.inf, -np.inf], np.nan)
    return out
