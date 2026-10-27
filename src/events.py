"""Economic calendar: release surprises and scheduled-event flags.

Source: the Forex Factory calendar CSVs (one per year, scraped with
``forex_factory_calendar_<year>.csv`` names). They are not in the repository;
``scripts/build_event_features.py`` turns them into
``data/events/event_surprises.csv``, which holds only event names,
UTC timestamps and standardised surprises.

Timezones. The CSV times are in the timezone of the browser that scraped
them: UTC+8 (Asia/Singapore, no daylight saving) for 2010-2023 and
Europe/Brussels from 2024. ``check_timezones`` verifies this against the
known release times of US payrolls (8:30 New York) and FOMC statements
(12:00-14:30 New York) before anything is converted. Up to 2012 the
calendar records the actual minute of release (e.g. 14:13).

Daily alignment. Gold and silver settle at 13:30 New York, copper at 13:00.
For the row of trading day t (target: the return from settle t to settle
t+1):

* ``z_*`` surprises are those released after settle t-1 and up to settle t,
  i.e. already public when the forecast is made;
* ``next_*`` flags mark scheduled releases between settle t and settle t+1.
  Release dates are published in advance. Only FOMC statements released
  between 12:00 and 14:30 New York count as scheduled (14:00 since 2013,
  14:15 before, 12:30 on the 2011-12 press-conference days), so the
  unscheduled cuts of 3 and 15 March 2020 are not flagged ahead of time.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

EVENTS_FILE = Path(__file__).resolve().parents[1] / "data" / "events" / "event_surprises.csv"
NEW_YORK = "America/New_York"

# Settlement time of the front-month futures (New York time).
SETTLE_TIME = {"gold": "13:30", "silver": "13:30", "copper": "13:00"}

SCRAPE_TZ = {**{y: "Asia/Singapore" for y in range(2010, 2024)}, **{y: "Europe/Brussels" for y in range(2024, 2027)}}

# (currency, event name) -> group used for the features.
GROUPS = {
    ("USD", "CPI m/m"): "cpi",
    ("USD", "Core CPI m/m"): "cpi",
    ("USD", "CPI y/y"): "cpi",
    ("USD", "Non-Farm Employment Change"): "nfp",
    ("USD", "Federal Funds Rate"): "fed_rate",
    ("CNY", "Manufacturing PMI"): "china_pmi",
    ("CNY", "Caixin Manufacturing PMI"): "china_pmi",
    ("CNY", "Caixin Flash Manufacturing PMI"): "china_pmi",
    ("CNY", "Non-Manufacturing PMI"): "china_pmi",
}
SURPRISE_GROUPS = ["cpi", "nfp", "fed_rate", "usd_other", "china_pmi"]
FLAG_GROUPS = ["cpi", "nfp", "fomc"]

_UNITS = {"K": 1e3, "k": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
_NUM = re.compile(r"^\s*<?\s*(-?\d+(?:\.\d+)?)\s*([%KkMBT]?)\s*$")


def parse_value(text) -> float:
    """'0.3%' -> 0.3, '250K' -> 250000, '-40.5B' -> -4.05e10. Votes ('9|0') etc. -> NaN."""
    if not isinstance(text, str):
        return np.nan
    m = _NUM.match(text)
    if not m:
        return np.nan
    value, unit = float(m.group(1)), m.group(2)
    return value * _UNITS.get(unit, 1.0)


def read_raw_calendar(raw_dir: str | Path, years: range | None = None) -> pd.DataFrame:
    """Read the yearly CSVs, keep timed rows, convert to UTC."""
    frames = []
    for path in sorted(Path(raw_dir).glob("forex_factory_calendar_*.csv")):
        year = int(re.search(r"(\d{4})\.csv$", path.name).group(1))
        if years is not None and year not in years:
            continue
        if year not in SCRAPE_TZ:
            raise ValueError(f"No verified scrape timezone for {year}; add it to SCRAPE_TZ after checking")
        df = pd.read_csv(path, dtype=str)
        timed = df["Time"].str.fullmatch(r"\d{1,2}:\d{2}(am|pm)", na=False)
        df = df[timed].copy()
        # 'Combined DateTime' carries the correct date also for the scraper's
        # 'Mon Feb 29' rows in leap years.
        date = pd.to_datetime(df["Combined DateTime"]).dt.strftime("%Y-%m-%d")
        local = pd.to_datetime(date + " " + df["Time"], format="%Y-%m-%d %I:%M%p")
        df["ts_utc"] = local.dt.tz_localize(SCRAPE_TZ[year], ambiguous="raise", nonexistent="raise").dt.tz_convert(
            "UTC"
        )
        df["file_year"] = year
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No forex_factory_calendar_<year>.csv files in {raw_dir}")
    cal = pd.concat(frames, ignore_index=True)
    cal["actual"] = cal["Actual"].map(parse_value)
    cal["forecast"] = cal["Forecast"].map(parse_value)
    cal = cal.drop_duplicates(subset=["ts_utc", "Currency", "Event"])
    return cal.sort_values("ts_utc").reset_index(drop=True)


def _scheduled_fomc_time(hhmm: pd.Series) -> pd.Series:
    return (hhmm >= "12:00") & (hhmm <= "14:30")


def check_timezones(cal: pd.DataFrame) -> pd.DataFrame:
    """Check the UTC conversion against fixed New York release times.

    Payrolls are released at 8:30 New York; scheduled FOMC statements fall
    between 12:00 and 14:30. Returns the share of matching releases per year
    and raises if a year falls below 95% (payrolls) or 75% (FOMC, which has
    unscheduled meetings).
    """
    ny = cal["ts_utc"].dt.tz_convert(NEW_YORK)
    hhmm = ny.dt.strftime("%H:%M")
    usd = cal["Currency"] == "USD"
    nfp = usd & (cal["Event"] == "Non-Farm Employment Change")
    fomc = usd & (cal["Event"] == "FOMC Statement")
    rows = []
    for year, g in cal.groupby("file_year"):
        n_ok = (hhmm[nfp & (cal.file_year == year)] == "08:30").mean()
        f_ok = _scheduled_fomc_time(hhmm[fomc & (cal.file_year == year)]).mean()
        rows.append({"year": year, "nfp_at_0830": n_ok, "fomc_at_1400": f_ok})
    out = pd.DataFrame(rows)
    bad = out[(out.nfp_at_0830 < 0.95) | (out.fomc_at_1400 < 0.75)]
    if len(bad):
        raise ValueError(f"Timezone check failed:\n{bad}")
    return out


def event_surprises(cal: pd.DataFrame, min_history: int = 12) -> pd.DataFrame:
    """One row per release: group and standardised surprise.

    z = (actual - forecast) / std of the *earlier* surprises of the same event
    (expanding window, at least ``min_history`` of them), so no release is
    scaled with information from its own future. Scheduled FOMC statements
    are kept as group 'fomc' for the flags.
    """
    cal = cal.copy()
    key = list(zip(cal["Currency"], cal["Event"], strict=True))
    cal["group"] = [GROUPS.get(k) for k in key]
    other = (cal["Currency"] == "USD") & (cal["Impact"] == "High") & cal["group"].isna()
    cal.loc[other, "group"] = "usd_other"
    ny_hhmm = cal["ts_utc"].dt.tz_convert(NEW_YORK).dt.strftime("%H:%M")
    regular_fomc = (cal["Currency"] == "USD") & (cal["Event"] == "FOMC Statement") & _scheduled_fomc_time(ny_hhmm)
    cal.loc[regular_fomc, "group"] = "fomc"
    cal = cal[cal["group"].notna()].copy()

    cal["surprise"] = cal["actual"] - cal["forecast"]
    z = pd.Series(np.nan, index=cal.index)
    for _, g in cal.groupby(["Currency", "Event"]):
        s = g["surprise"].dropna()
        if s.empty:
            continue
        past_sd = s.expanding().std().shift(1)
        past_n = s.expanding().count().shift(1)
        zz = s / past_sd
        zz[(past_n < min_history) | (past_sd <= 0)] = np.nan
        z.loc[zz.index] = zz.clip(-5, 5)
    cal["z"] = z
    return cal[["ts_utc", "Currency", "Event", "group", "z"]].rename(
        columns={"Currency": "currency", "Event": "event"}
    )


def settlement_times(dates: pd.DatetimeIndex, commodity: str) -> pd.DatetimeIndex:
    """UTC timestamps of the daily settlement on each trading date."""
    hhmm = SETTLE_TIME[commodity]
    local = pd.to_datetime(dates.strftime("%Y-%m-%d") + " " + hhmm).tz_localize(NEW_YORK)
    return local.tz_convert("UTC")


def daily_event_features(dates: pd.DatetimeIndex, commodity: str, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Event features for each trading date (see module docstring)."""
    if events is None:
        if not EVENTS_FILE.exists():
            raise FileNotFoundError(f"{EVENTS_FILE} missing; run scripts/build_event_features.py")
        events = pd.read_csv(EVENTS_FILE, parse_dates=["ts_utc"])
    ts = pd.to_datetime(events["ts_utc"], utc=True)
    settle = settlement_times(dates, commodity)
    # Index of the first settlement at or after each event: the event is
    # public by the close of that trading day.
    settle_ns = settle.as_unit("ns").asi8
    event_ns = ts.dt.as_unit("ns").astype("int64").to_numpy()
    pos = np.searchsorted(settle_ns, event_ns, side="left")
    inside = pos < len(dates)
    ev = events.assign(pos=pos)[inside]

    out = pd.DataFrame(index=dates)
    for group in SURPRISE_GROUPS:
        g = ev[(ev.group == group) & ev.z.notna()]
        mean_z = g.groupby("pos")["z"].mean()
        col = np.zeros(len(dates))
        col[mean_z.index.to_numpy()] = mean_z.to_numpy()
        out[f"z_{group}"] = col
    # Scheduled releases inside (settle t, settle t+1] belong to row t, i.e.
    # one row before the row at which they become public.
    for group in FLAG_GROUPS:
        p = ev.loc[ev.group == group, "pos"].unique() - 1
        col = np.zeros(len(dates))
        col[p[p >= 0]] = 1.0
        out[f"next_{group}"] = col
    first_event = ts.min()
    out["calendar_covered"] = (settle >= first_event).astype(float)
    return out
