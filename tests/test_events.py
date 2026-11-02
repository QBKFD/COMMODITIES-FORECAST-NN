import numpy as np
import pandas as pd
import pytest

from src import events
from src.train_tf import prepare_frame


def write_calendar(tmp_path, year, rows):
    df = pd.DataFrame(
        rows, columns=["Date", "Time", "Currency", "Impact", "Event", "Actual", "Forecast", "Previous"]
    )
    df["Combined DateTime"] = df["Date"] + " " + df["Time"]
    df.to_csv(tmp_path / f"forex_factory_calendar_{year}.csv", index=False)


def test_parse_value():
    assert events.parse_value("0.3%") == pytest.approx(0.3)
    assert events.parse_value("250K") == pytest.approx(250_000)
    assert events.parse_value("-40.5B") == pytest.approx(-4.05e10)
    assert events.parse_value("<0.25%") == pytest.approx(0.25)
    assert np.isnan(events.parse_value("9|0"))
    assert np.isnan(events.parse_value(None))


def test_scrape_timezone_conversion(tmp_path):
    # 2015 file is in UTC+8: payrolls at 8:30 New York (EST) = 21:30 Singapore.
    write_calendar(
        tmp_path,
        2015,
        [
            ["2015-01-09", "9:30pm", "USD", "High", "Non-Farm Employment Change", "252K", "241K", "353K"],
            ["2015-01-28", "3:00am", "USD", "High", "FOMC Statement", "", "", ""],
            ["2015-01-28", "All Day", "USD", "Holiday", "Bank Holiday", "", "", ""],
        ],
    )
    cal = events.read_raw_calendar(tmp_path)
    assert len(cal) == 2  # untimed row dropped
    ny = cal["ts_utc"].dt.tz_convert(events.NEW_YORK).dt.strftime("%Y-%m-%d %H:%M").tolist()
    assert ny == ["2015-01-09 08:30", "2015-01-27 14:00"]
    events.check_timezones(cal)


def test_wrong_timezone_is_caught(tmp_path):
    # The same release written at 8:30 local time would be 19:30 New York the day before.
    write_calendar(
        tmp_path, 2015, [["2015-01-09", "8:30am", "USD", "High", "Non-Farm Employment Change", "1K", "1K", ""]]
    )
    with pytest.raises(ValueError, match="Timezone"):
        events.check_timezones(events.read_raw_calendar(tmp_path))


def _cal(surprises, start="2015-01-01"):
    ts = pd.date_range(start, periods=len(surprises), freq="MS", tz="UTC") + pd.Timedelta(hours=13, minutes=30)
    return pd.DataFrame(
        {
            "ts_utc": ts,
            "Currency": "USD",
            "Event": "CPI m/m",
            "Impact": "High",
            "actual": np.asarray(surprises, float),
            "forecast": 0.0,
        }
    )


def test_surprise_scaled_by_past_only():
    rng = np.random.default_rng(0)
    s = rng.normal(size=30)
    z1 = events.event_surprises(_cal(s), min_history=12)["z"].to_numpy()
    s2 = s.copy()
    s2[20:] *= 100  # changing later surprises must not change earlier scores
    z2 = events.event_surprises(_cal(s2), min_history=12)["z"].to_numpy()
    assert np.all(np.isnan(z1[:12]))
    np.testing.assert_allclose(z1[12:20], z2[12:20])
    assert z1[15] == pytest.approx(s[15] / np.std(s[:15], ddof=1))


def test_daily_alignment():
    dates = pd.bdate_range("2020-01-06", "2020-01-10")  # Mon-Fri, EST (UTC-5)
    ev = pd.DataFrame(
        {
            "ts_utc": pd.to_datetime(
                [
                    "2020-01-07 13:30",  # Tue 8:30 ET, before settle -> public on Tue
                    "2020-01-08 19:00",  # Wed 14:00 ET, after gold settle -> public on Thu
                    "2020-01-08 18:15",  # Wed 13:15 ET: after copper (13:00), before gold (13:30)
                ],
                utc=True,
            ),
            "group": ["nfp", "fomc", "cpi"],
            "z": [1.5, np.nan, -2.0],
        }
    )
    gold = events.daily_event_features(dates, "gold", ev)
    copper = events.daily_event_features(dates, "copper", ev)
    assert gold["z_nfp"].tolist() == [0, 1.5, 0, 0, 0]
    # The FOMC statement falls in Wed->Thu, so the flag is on Wednesday's row.
    assert gold["next_fomc"].tolist() == [0, 0, 1, 0, 0]
    assert gold["next_nfp"].tolist() == [1, 0, 0, 0, 0]
    assert gold["z_cpi"].tolist() == [0, 0, -2.0, 0, 0]
    assert copper["z_cpi"].tolist() == [0, 0, 0, -2.0, 0]


def test_unscheduled_fomc_not_flagged():
    cal = pd.DataFrame(
        {
            "ts_utc": pd.to_datetime(["2020-01-29 19:00", "2020-03-03 15:00"], utc=True),  # 14:00 and 10:00 ET
            "Currency": "USD",
            "Event": "FOMC Statement",
            "Impact": "High",
            "actual": np.nan,
            "forecast": np.nan,
        }
    )
    out = events.event_surprises(cal)
    assert out.loc[out["group"] == "fomc", "ts_utc"].dt.month.tolist() == [1]


def test_train_start_drops_early_rows():
    idx = pd.bdate_range("2009-12-01", periods=60)
    df = pd.DataFrame({"a": np.arange(60.0), "b": 1.0, "target": 0.01}, index=idx)
    frame = prepare_frame(df, ["a"], ["b"], train_start="2010-01-01")
    assert frame.index.min() >= pd.Timestamp("2010-01-01")
    assert len(prepare_frame(df, ["a"], ["b"])) == 60
