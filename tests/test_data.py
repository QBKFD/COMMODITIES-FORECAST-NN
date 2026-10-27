import numpy as np
import pandas as pd
import pytest

from src.data import RELEASE_LAG_DAYS, apply_release_lag, load_commodity, make_windows


def test_release_lag_delays_monthly_values():
    days = pd.bdate_range("2021-01-01", "2021-05-31")
    month_start = days.to_period("M").start_time
    raw = pd.Series(month_start.month.astype(float), index=days)  # value = month number
    lagged = apply_release_lag(raw, 45)
    # March's value (3) must not be visible before 1 March + 45 days.
    assert (lagged[lagged == 3].index.min()) >= pd.Timestamp("2021-04-15")
    assert lagged.loc["2021-04-14"] == 2
    # Never shows a value earlier than the raw data does.
    assert (lagged.dropna() <= raw[lagged.dropna().index]).all()


@pytest.mark.parametrize("name", ["gold", "silver", "copper"])
def test_loaded_data_has_no_early_macro_values(name):
    """On day d a macro series shows the raw value as of d - lag: nothing newer."""
    raw = pd.read_csv(f"data/{name}_fixed_auto.csv", parse_dates=["ds"], thousands=",").set_index("ds")
    raw = raw[~raw.index.duplicated()]
    df = load_commodity(name)
    for col, lag in RELEASE_LAG_DAYS.items():
        if col not in raw:
            continue
        r = raw[col].dropna()
        expected = r.reindex(df.index - pd.Timedelta(days=lag), method="ffill").to_numpy()
        got = df[col].to_numpy()
        known = ~np.isnan(expected) & ~np.isnan(got)
        assert known.mean() > 0.95, col
        assert np.allclose(got[known], expected[known]), col


def test_target_is_next_day_log_return():
    df = load_commodity("gold")
    close = df["Close"]
    expected = np.log(close.shift(-1) / close)
    assert np.allclose(df["target"].dropna(), expected.dropna().loc[df["target"].dropna().index])


def test_external_markets_are_lagged_one_day():
    raw = pd.read_csv("data/gold_fixed_auto.csv", parse_dates=["ds"], thousands=",").set_index("ds")
    df = load_commodity("gold")
    assert df.loc["2021-03-02", "SP500_log_return"] == pytest.approx(raw.loc["2021-03-01", "SP500_log_return"])


def test_windows_end_on_forecast_day():
    feats = np.arange(10, dtype=float).reshape(-1, 1)
    target = np.arange(10, dtype=float) * 10
    X, y, ends = make_windows(feats, target, lookback=3)
    assert X.shape == (8, 3, 1)
    assert X[0, :, 0].tolist() == [0, 1, 2] and y[0] == 20 and ends[0] == 2
    assert X[-1, -1, 0] == 9 and y[-1] == 90


def test_stationary_features_have_no_look_ahead():
    """Features on day t must not change when later prices change."""
    import numpy as np
    import pandas as pd

    from src.data import load_commodity
    from src.features import MACRO_FEATURES, PRICE_FEATURES

    df = load_commodity("gold")
    cols = PRICE_FEATURES + MACRO_FEATURES
    cut = df.index[3000]
    from src.features import add_features

    # add_features recomputes every derived column from the raw ones
    full = add_features(df)
    early = add_features(df.loc[:cut])
    pd.testing.assert_frame_equal(full.loc[:cut, cols], early[cols])
    assert not np.isinf(full[cols].to_numpy(float)).any()


def test_features_are_stable_across_decades():
    """Price-level columns drift with the price; the model inputs should not."""
    from src.data import load_commodity
    from src.features import PRICE_FEATURES

    f = load_commodity("gold")[PRICE_FEATURES].dropna()
    train, test = f.loc[:"2019"], f.loc["2024"]
    z = ((test.mean() - train.mean()) / train.std()).abs()
    assert z.max() < 1.5, z.sort_values().tail(3)
