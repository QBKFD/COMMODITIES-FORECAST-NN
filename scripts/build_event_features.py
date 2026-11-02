"""Build data/events/event_surprises.csv from the Forex Factory calendar exports.

    python scripts/build_event_features.py --raw data/calendar_raw

``--raw`` must hold the yearly ``forex_factory_calendar_<year>.csv`` files.
They are not part of the repository (data/calendar_raw/ is ignored by git);
the output only contains event names, UTC release times and standardised
surprises. The script stops if the timezone check in ``src/events.py`` fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.events import EVENTS_FILE, check_timezones, event_surprises, read_raw_calendar  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, default=ROOT / "data" / "calendar_raw")
    p.add_argument("--first-year", type=int, default=2010)
    p.add_argument("--last-year", type=int, default=2024, help="the commodity data end in November 2024")
    p.add_argument("--out", type=Path, default=EVENTS_FILE)
    args = p.parse_args()

    cal = read_raw_calendar(args.raw, range(args.first_year, args.last_year + 1))
    checks = check_timezones(cal)
    print("Timezone check (share of releases at the expected New York time):")
    print(checks.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    events = event_surprises(cal)
    events = events.assign(ts_utc=events["ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"), z=events["z"].round(4))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.out, index=False)

    print(f"\n{len(events)} releases written to {args.out}")
    print(events.groupby("group")["z"].agg(releases="size", with_surprise="count").to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
