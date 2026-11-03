"""Time base (plan §3.1): absolute Unix seconds (UTC) address all content.

Conversions to Skyfield time, ISO 8601 strings and durations. No network access: Skyfield's built-in leap-second
and delta-T tables are used.
"""

import datetime as dt
import re
from functools import cache
from typing import cast

import numpy as np
from skyfield.api import Time, Timescale, load
from skyfield.nutationlib import iau2000b_radians

_EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)
_US = dt.timedelta(microseconds=1)
_UNITS = {"ms": 1e-3, "s": 1.0, "sec": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0}
_DURATION = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(ms|sec|s|min|h|d)?\s*$")


@cache
def timescale() -> Timescale:
    return cast("Timescale", load.timescale())  # built-in leap-second / delta-T tables, no download


def unix_to_time(t_unix: float | np.ndarray) -> Time:
    """Skyfield Time from Unix seconds (UTC).

    Not ts.utc(1970, 1, 1, 0, 0, t): that applies the 1970 leap-second offset and is 27 s off in 2026.
    Splitting into whole days + seconds of day makes Skyfield look up leap seconds on the right date.
    """
    t = np.asarray(t_unix, dtype=np.float64)
    days = np.floor(t / 86400.0)
    return cast("Time", timescale().utc(1970, 1, 1 + days.astype(np.int64), 0, 0, t - days * 86400.0))


def unix_to_time_fast(t_unix: float | np.ndarray) -> Time:
    """As unix_to_time, with the IAU 2000B nutation model (Skyfield's documented speed-up).

    8x faster than the full IAU 2000A series. Station-to-satellite geometry changes by ~1e-8 m, because both ends
    rotate with the same nutation. Used by the simulator's geometry sampling; the profile tool keeps the full model.
    """
    t = unix_to_time(t_unix)
    t._nutation_angles_radians = iau2000b_radians(t)
    return t


def parse_iso(s: str) -> float:
    """Unix seconds from an ISO 8601 time with an explicit zone (`Z` or `+hh:mm`), exact to 1 µs."""
    d = dt.datetime.fromisoformat(s.strip())
    if d.tzinfo is None:
        raise ValueError(f"time {s!r} has no zone; add 'Z' for UTC")
    return ((d - _EPOCH) // _US) / 1e6


def format_iso(t_unix: float, decimals: int | None = None) -> str:
    """ISO 8601 UTC with a `Z`. decimals=None: whole seconds, or microseconds when there is a fraction."""
    us = round(t_unix * 1e6)
    d = _EPOCH + dt.timedelta(microseconds=us)
    if decimals is None:
        return d.strftime("%Y-%m-%dT%H:%M:%S") + (f".{d.microsecond:06d}" if us % 1_000_000 else "") + "Z"
    if decimals == 0:
        return (_EPOCH + dt.timedelta(seconds=round(t_unix))).strftime("%Y-%m-%dT%H:%M:%SZ")
    q = round(t_unix * 10**decimals)
    whole, frac = divmod(q, 10**decimals)
    return (_EPOCH + dt.timedelta(seconds=whole)).strftime("%Y-%m-%dT%H:%M:%S") + f".{frac:0{decimals}d}Z"


def parse_duration(v: str | float) -> float:
    """Seconds from a number or a string like `90`, `30s`, `250ms`, `10min`, `2h`, `1d`."""
    if isinstance(v, bool):
        raise ValueError(f"not a duration: {v!r}")
    if isinstance(v, int | float):
        return float(v)
    m = _DURATION.match(v)
    if not m:
        raise ValueError(f"not a duration: {v!r} (use e.g. 90, 30s, 10min, 2h, 1d)")
    return float(m.group(1)) * _UNITS[m.group(2) or "s"]


def parse_time(s: str | float, ref_unix: float) -> float:
    """A point in time: ISO 8601, `start` (= ref), or an offset from ref such as `+2min` or `-30s`."""
    if isinstance(s, int | float) and not isinstance(s, bool):
        return float(s)
    s = str(s).strip()
    if s == "start":
        return ref_unix
    if s[:1] in "+-":
        return ref_unix + parse_duration(s)
    return parse_iso(s)
