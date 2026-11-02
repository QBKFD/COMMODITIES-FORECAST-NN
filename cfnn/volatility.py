"""Next-day variance forecasts for gold, silver and copper.

Design: ``docs/volatility_design.md`` (written before any model was run).

Every model forecasts h(t+1), the variance of the next day's log return,
from data up to and including day t. The target is the squared next-day
return, r²(t+1). Models are refit once per test year on data from
``START`` to the end of the previous year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cfnn.data import load_commodity, make_windows
from cfnn.events import add_event_features

START = "2010-03-01"
TEST_YEARS = [2020, 2021, 2022, 2023, 2024]
FLOOR = 1e-8  # a daily volatility of 0.01%
EWMA_LAMBDA = 0.94
FLAGS = ["next_cpi", "next_nfp", "next_fomc"]
HAR_COLS = ["r2_1", "r2_5", "r2_22"]
HARX_COLS = HAR_COLS + ["range2", "vix2"] + FLAGS
NN_INPUTS = [
    "ret", "abs_r", "hl_range", "rolling_vol_5", "rolling_vol_20", "atr_pct",
    "VIX_Close", "VIX_change",
] + FLAGS  # fmt: skip
LOOKBACK = 22
CLIP = 5.0


# Data ---------------------------------------------------------------------------


def load(commodity: str) -> pd.DataFrame:
    """Daily frame from START with the target ``y`` and all model inputs."""
    d = add_event_features(load_commodity(commodity), commodity)
    # Returns from the closing prices, so that r²(t) is exactly the target of day t-1.
    # (The precomputed log_return column disagrees with Close on one gold row, 2013-09-02.)
    d["ret"] = np.log(d["Close"] / d["Close"].shift(1))
    d["abs_r"] = d["ret"].abs()
    d = d.loc[START:].copy()
    r2 = d["ret"] ** 2
    d["r2_1"] = r2
    d["r2_5"] = r2.rolling(5).mean()
    d["r2_22"] = r2.rolling(22).mean()
    d["range2"] = d["hl_range"] ** 2
    vix = d["VIX_Close"].ffill()
    d["vix2"] = (vix / 100) ** 2 / 252  # VIX as a daily variance (it is already lagged one day)
    d["y"] = d["target"] ** 2
    d["release_day"] = (d[FLAGS].max(axis=1) > 0).astype(int)
    d = d[d["target"].notna()]
    return d


def train_test_masks(index: pd.DatetimeIndex, year: int) -> tuple[np.ndarray, np.ndarray]:
    """Training rows: target realised before the test year. Test rows: forecasts made in it."""
    next_day = pd.Series(index, index=index).shift(-1)
    train = (next_day < pd.Timestamp(f"{year}-01-01")).to_numpy()
    test = np.asarray(index.year == year)
    return train, test


# Losses -------------------------------------------------------------------------


def qlike(y, h) -> np.ndarray:
    """QLIKE per day, r²/h + log h (Patton 2011, up to a constant)."""
    h = np.maximum(np.asarray(h, float), FLOOR)
    return np.asarray(y, float) / h + np.log(h)


# Benchmarks ---------------------------------------------------------------------


def ewma(r: np.ndarray, lam: float = EWMA_LAMBDA, init_n: int = 22) -> np.ndarray:
    """RiskMetrics: forecast made on day t for day t+1, from returns up to t."""
    out = np.empty(len(r))
    s2 = float(np.mean(r[:init_n] ** 2))
    for t, x in enumerate(r):
        s2 = lam * s2 + (1 - lam) * x * x
        out[t] = s2
    return out


def ols_forecast(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> np.ndarray:
    tr = train.dropna(subset=cols + ["y"])
    X = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy() for c in cols])
    beta = np.linalg.lstsq(X, tr["y"].to_numpy(), rcond=None)[0]
    Xt = np.column_stack([np.ones(len(test))] + [test[c].ffill().to_numpy() for c in cols])
    raw = Xt @ beta
    # Insanity filter (see the amendment in docs/volatility_design.md): a linear
    # model can forecast a negative variance; use the training mean instead.
    return np.maximum(np.where(raw > 0, raw, tr["y"].mean()), FLOOR)


def garch_forecast(d: pd.DataFrame, year: int, gjr: bool) -> np.ndarray:
    """GARCH(1,1) or GJR-GARCH(1,1,1), Student-t, zero mean; fit before ``year``, filter through it."""
    from arch import arch_model

    r = 100 * d["ret"]
    first_test = d.index[d.index.year == year][0]
    am = arch_model(r, mean="Zero", vol="GARCH", p=1, o=1 if gjr else 0, q=1, dist="t")
    res = am.fit(last_obs=first_test, disp="off")
    fc = res.forecast(horizon=1, start=first_test, reindex=False)
    h = fc.variance.iloc[:, 0]
    h = h.loc[d.index[d.index.year == year]]
    return np.maximum(h.to_numpy() / 1e4, FLOOR)


def benchmark_forecasts(d: pd.DataFrame, model: str) -> pd.DataFrame:
    """Walk-forward forecasts of one benchmark model over all test years."""
    ew = ewma(d["ret"].fillna(0).to_numpy()) if model == "ewma" else None
    parts = []
    for year in TEST_YEARS:
        train, test = train_test_masks(d.index, year)
        tr, te = d[train], d[test]
        if model == "hist_mean":
            h = np.full(len(te), tr["y"].mean())
        elif model == "ewma":
            h = ew[test]
        elif model == "har":
            h = ols_forecast(tr, te, HAR_COLS)
        elif model == "har_x":
            h = ols_forecast(tr, te, HARX_COLS)
        elif model in ("garch", "gjr_garch"):
            h = garch_forecast(d, year, gjr=model == "gjr_garch")
        else:
            raise ValueError(model)
        parts.append(pd.DataFrame({"date": te.index, "year": year, "y": te["y"].to_numpy(),
                                   "h": h, "release_day": te["release_day"].to_numpy()}))  # fmt: skip
    return pd.concat(parts, ignore_index=True)


# Neural networks ----------------------------------------------------------------


def build_network(kind: str, n_features: int):
    import tensorflow as tf
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    from cfnn.architectures.blocks import LastStep, embed, encoder_block

    inp = Input(shape=(LOOKBACK, n_features))
    if kind == "lstm":
        x = LSTM(32)(inp)
    elif kind == "transformer":
        x = embed(inp, 32, LOOKBACK)
        for _ in range(2):
            x = encoder_block(x, heads=4, key_dim=8, ff_dim=64, dropout=0.1)
        x = LastStep()(x)
    else:
        raise ValueError(kind)
    x = Dropout(0.1)(Dense(16, activation="gelu")(x))
    out = Dense(1, name="log_h")(x)
    model = Model(inp, out)

    def qlike_loss(y_true, z):
        # z = log h (in units of the training-window mean variance)
        return tf.reduce_mean(z + y_true * tf.exp(-z))

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4, clipnorm=1.0),
        loss=qlike_loss,
    )
    return model


def network_forecasts(d: pd.DataFrame, kind: str, seed: int) -> pd.DataFrame:
    import tensorflow as tf

    raw = d[NN_INPUTS].ffill().fillna(0).to_numpy(float)
    y_all = d["y"].to_numpy(float)
    parts = []
    for year in TEST_YEARS:
        train, test = train_test_masks(d.index, year)
        last_train = int(np.flatnonzero(train)[-1])
        mu, sd = raw[: last_train + 1].mean(0), raw[: last_train + 1].std(0)
        sd[sd == 0] = 1.0
        X = np.clip((raw - mu) / sd, -CLIP, CLIP)
        scale = float(y_all[: last_train + 1].mean())
        Xw, yw, ends = make_windows(X, y_all / scale, LOOKBACK)
        tr_k = ends <= last_train
        te_k = test[ends]
        Xtr, ytr = Xw[tr_k], yw[tr_k]
        n_val = int(round(0.15 * len(Xtr)))

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        model = build_network(kind, X.shape[1])
        model.fit(
            Xtr[:-n_val], ytr[:-n_val],
            validation_data=(Xtr[-n_val:], ytr[-n_val:]),
            epochs=60, batch_size=64, verbose=0, shuffle=True,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)],
        )  # fmt: skip
        z = model.predict(Xw[te_k], verbose=0).ravel()
        h = np.maximum(scale * np.exp(z), FLOOR)
        rows = ends[te_k]
        parts.append(pd.DataFrame({"date": d.index[rows], "year": year, "y": y_all[rows], "h": h,
                                   "release_day": d["release_day"].to_numpy()[rows]}))  # fmt: skip
    return pd.concat(parts, ignore_index=True)
