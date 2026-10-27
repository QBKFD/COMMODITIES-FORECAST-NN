"""Walk-forward training and evaluation for the Keras models and the benchmarks.

For every test year Y:

* training data are all windows whose target (the next-day return) is
  realised before 1 January of Y;
* the last ``val_fraction`` of those windows (in time order) is held out for
  early stopping, and the scalers are fitted on the rest only;
* the model forecasts every trading day of Y, one day ahead, from windows
  that end on the forecast day.

Predictions are returned per day together with the naive baselines, so the
evaluation can compare them on exactly the same days.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data import TARGET, make_windows

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

NN_MODELS = {
    "stacked_lstm": "tf_stacked_lstm",
    "stacked_gru": "tf_stacked_gru",
    "lstm_attention": "tf_tft_lstm",
    "gru_attention": "tf_tft_gru",
    "transformer": "tf_transformers",
}
CLIP = 5.0
BENCHMARK_MODELS = ("xgboost", "random_forest", "svr")

# Names used by the original scripts and configs.
ALIASES = {
    "tf_stacked_lstm": "stacked_lstm",
    "tf_stacked_gru": "stacked_gru",
    "tf_hybrid_lstm": "lstm_attention",
    "tf_tft_lstm": "lstm_attention",
    "tf_tft_gru": "gru_attention",
    "tf_transformers": "transformer",
}


def resolve_model_name(name: str, config: dict) -> str:
    if name == "benchmarks":
        name = config.get("benchmark_type", "xgboost").lower()
    name = ALIASES.get(name, name)
    if name not in NN_MODELS and name not in BENCHMARK_MODELS:
        known = sorted([*NN_MODELS, *BENCHMARK_MODELS])
        raise KeyError(f"Unknown model {name!r}; choose from {known}")
    return name


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:  # noqa: BLE001 - not available on every build
        pass


@dataclass
class Fold:
    year: int
    Xp_fit: np.ndarray
    Xm_fit: np.ndarray
    y_fit: np.ndarray
    Xp_val: np.ndarray
    Xm_val: np.ndarray
    y_val: np.ndarray
    Xp_test: np.ndarray
    Xm_test: np.ndarray
    y_test: np.ndarray  # scaled
    y_test_raw: np.ndarray  # log returns
    y_train_raw: np.ndarray
    test_dates: pd.DatetimeIndex
    target_scaler: StandardScaler


def prepare_frame(
    df: pd.DataFrame,
    price_features: list[str],
    macro_features: list[str],
    train_start: str | None = None,
) -> pd.DataFrame:
    """Keep the needed columns, carry the last known value over gaps, drop the warm-up.

    ``train_start`` drops all earlier rows (used when a feature, such as the
    event calendar, only exists from a later date).
    """
    missing = [c for c in [*price_features, *macro_features] if c not in df]
    if missing:
        raise KeyError(f"Columns not in the data: {missing}")
    cols = list(dict.fromkeys([*price_features, *macro_features]))
    out = df[cols].ffill()  # forward fill only: never uses later values
    out[TARGET] = df[TARGET]
    out = out.dropna(subset=cols)
    if train_start is not None:
        out = out[out.index >= pd.Timestamp(train_start)]
    return out.iloc[:-1] if np.isnan(out[TARGET].iloc[-1]) else out


def make_fold(
    frame: pd.DataFrame,
    price_features: list[str],
    macro_features: list[str],
    lookback: int,
    year: int,
    val_fraction: float = 0.15,
) -> Fold | None:
    dates = frame.index
    test_start = pd.Timestamp(f"{year}-01-01")
    test_end = pd.Timestamp(f"{year + 1}-01-01")
    first_test = int(np.searchsorted(dates, test_start))
    if first_test == 0 or first_test >= len(dates):
        return None

    # Rows whose target is realised before the test year: the target on row p
    # is the return to row p+1, so the last usable training row is first_test-2.
    last_train_row = first_test - 2
    n_train_windows = last_train_row - (lookback - 1) + 1
    if n_train_windows < 50:
        return None
    n_val = max(1, int(round(n_train_windows * val_fraction)))
    last_fit_row = last_train_row - n_val

    price_raw = frame[price_features].to_numpy(float)
    macro_raw = frame[macro_features].to_numpy(float)
    target_raw = frame[TARGET].to_numpy(float)

    # Scalers see only the rows used for fitting.
    sp = StandardScaler().fit(price_raw[: last_fit_row + 1])
    sm = StandardScaler().fit(macro_raw[: last_fit_row + 1])
    st = StandardScaler().fit(target_raw[lookback - 1 : last_fit_row + 1].reshape(-1, 1))
    # Clip at 5 standard deviations so a handful of crisis days (March 2020)
    # cannot dominate the fit.
    price = np.clip(sp.transform(price_raw), -CLIP, CLIP)
    macro = np.clip(sm.transform(macro_raw), -CLIP, CLIP)
    target = st.transform(target_raw.reshape(-1, 1)).ravel()

    Xp, y, ends = make_windows(price, target, lookback)
    Xm, _, _ = make_windows(macro, target, lookback)
    in_test = (dates[ends] >= test_start) & (dates[ends] < test_end)
    fit = ends <= last_fit_row
    val = (ends > last_fit_row) & (ends <= last_train_row)
    if not in_test.any():
        return None

    return Fold(
        year=year,
        Xp_fit=Xp[fit], Xm_fit=Xm[fit], y_fit=y[fit],
        Xp_val=Xp[val], Xm_val=Xm[val], y_val=y[val],
        Xp_test=Xp[in_test], Xm_test=Xm[in_test], y_test=y[in_test],
        y_test_raw=target_raw[ends[in_test]],
        y_train_raw=target_raw[lookback - 1 : last_train_row + 1],
        test_dates=dates[ends[in_test]],
        target_scaler=st,
    )


def _fit_nn(name: str, config: dict, fold: Fold, verbose: int = 0):
    import importlib

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    module = importlib.import_module(f"src.architectures.{NN_MODELS[name]}")
    lookback = fold.Xp_fit.shape[1]
    model = module.build_model(config, lookback, fold.Xp_fit.shape[2], fold.Xm_fit.shape[2])
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=config.get("patience", 5), restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]
    model.fit(
        {"price_input": fold.Xp_fit, "macro_input": fold.Xm_fit},
        fold.y_fit,
        validation_data=({"price_input": fold.Xp_val, "macro_input": fold.Xm_val}, fold.y_val),
        epochs=config.get("epochs", 25),
        batch_size=config.get("batch_size", 32),
        shuffle=True,
        verbose=verbose,
        callbacks=callbacks,
    )
    pred = model.predict({"price_input": fold.Xp_test, "macro_input": fold.Xm_test}, verbose=0)
    return model, np.asarray(pred).reshape(-1)


def _fit_benchmark(name: str, config: dict, fold: Fold, seed: int):
    from src.architectures import benchmarks

    cfg = {**config, "benchmark_type": name, "random_state": seed}
    model = benchmarks.build_model(cfg)

    def flat(Xp, Xm):
        return np.concatenate([Xp, Xm], axis=2).reshape(len(Xp), -1)

    # Benchmarks have no early stopping, so they train on fit + validation windows.
    X_train = np.concatenate([flat(fold.Xp_fit, fold.Xm_fit), flat(fold.Xp_val, fold.Xm_val)])
    y_train = np.concatenate([fold.y_fit, fold.y_val])
    model.fit(X_train, y_train)
    return model, np.asarray(model.predict(flat(fold.Xp_test, fold.Xm_test))).reshape(-1)


def walk_forward(
    df: pd.DataFrame,
    model_name: str,
    config: dict,
    seed: int = 0,
    years: list[int] | None = None,
    verbose: int = 0,
    keep_models: bool = False,
) -> tuple[pd.DataFrame, list]:
    """Run the walk-forward evaluation for one model and seed.

    Returns a frame with one row per test day (date, year, y_true, y_pred and
    the naive baselines) and, if ``keep_models``, the fitted models with their
    folds (used for SHAP).
    """
    name = resolve_model_name(model_name, config)
    price_features = config["price_features"]
    macro_features = config["macro_features"]
    lookback = config.get("lookback", 30)
    years = years or config.get("years", [2020, 2021, 2022, 2023, 2024])
    if config.get("horizon", 1) != 1:
        raise ValueError("Only one-day-ahead forecasts (horizon 1) are supported by this pipeline")

    frame = prepare_frame(df, price_features, macro_features, config.get("train_start"))
    rows, kept = [], []
    for year in years:
        fold = make_fold(frame, price_features, macro_features, lookback, year, config.get("val_fraction", 0.15))
        if fold is None:
            continue
        set_seed(seed)
        if name in NN_MODELS:
            model, pred_scaled = _fit_nn(name, config, fold, verbose)
        else:
            model, pred_scaled = _fit_benchmark(name, config, fold, seed)
        pred = fold.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        rows.append(
            pd.DataFrame(
                {
                    "date": fold.test_dates,
                    "year": year,
                    "y_true": fold.y_test_raw,
                    "y_pred": pred,
                    "pred_mean": float(np.mean(fold.y_train_raw)),
                }
            )
        )
        if keep_models:
            kept.append((model, fold))
        # Free the Keras graph between folds.
        if name in NN_MODELS:
            import tensorflow as tf

            tf.keras.backend.clear_session()
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out["model"] = name
    out["seed"] = seed
    return out, kept
