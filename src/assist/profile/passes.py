"""Small pass-geometry helper for the profile tool (plan §9). Moves into orbits/ in M2.

Reads OMM CSV snapshots by header name, merges them by NORAD id (newer epoch wins), and gives passes and
topocentric geometry (elevation, azimuth, range, range rate) at Unix times. No network access: Skyfield's
built-in time scale files are used.
"""

import csv
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from skyfield.api import EarthSatellite, Time, Timescale, load, wgs84

if TYPE_CHECKING:
    from skyfield.toposlib import GeographicPosition

    from assist.profile.station import Station

C_M_S = 299_792_458.0

OMM_COLUMNS = (
    "OBJECT_NAME",
    "OBJECT_ID",
    "EPOCH",
    "MEAN_MOTION",
    "ECCENTRICITY",
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "MEAN_ANOMALY",
    "EPHEMERIS_TYPE",
    "CLASSIFICATION_TYPE",
    "NORAD_CAT_ID",
    "ELEMENT_SET_NO",
    "REV_AT_EPOCH",
    "BSTAR",
    "MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT",
)


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


def read_omm_csv(path: str | Path) -> list[dict[str, str]]:
    """Rows of an OMM CSV (retlector layout), checked by header name so upstream column changes fail loudly."""
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        missing = set(OMM_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing OMM columns {sorted(missing)}")
        return list(reader)


def load_elements(paths: list[str | Path]) -> dict[int, dict[str, str]]:
    """Union of element files by NORAD id; where several hold an object, the newest EPOCH wins."""
    out: dict[int, dict[str, str]] = {}
    for p in paths:
        for row in read_omm_csv(p):
            norad = int(row["NORAD_CAT_ID"])
            if norad not in out or row["EPOCH"] > out[norad]["EPOCH"]:  # ISO strings sort chronologically
                out[norad] = row
    return out


def satellite(row: dict[str, str]) -> EarthSatellite:
    return cast("EarthSatellite", EarthSatellite.from_omm(timescale(), row))


@dataclass(frozen=True)
class Pass:
    norad: int
    name: str
    aos: float  # Unix seconds (horizon crossing at min_el)
    tca: float  # culmination
    los: float
    max_el_deg: float


def find_passes(row: dict[str, str], station: Station, t0: float, t1: float, min_el_deg: float = 0.0) -> list[Pass]:
    """Passes above min_el overlapping [t0, t1]. Passes in progress at t0 or t1 are clipped to the window."""
    sat = satellite(row)
    topo = _station_position(station)
    times, events = sat.find_events(topo, unix_to_time(t0), unix_to_time(t1), altitude_degrees=min_el_deg)
    unix: list[float] = [float(d.timestamp()) for d in times.utc_datetime()] if len(events) else []
    passes: list[Pass] = []
    aos: float | None = t0 if _elevation(sat, topo, t0) > min_el_deg else None
    tca: float | None = None
    for t, e in zip(unix, events, strict=True):
        if e == 0:
            aos, tca = t, None
        elif e == 1:
            tca = t
        elif e == 2 and aos is not None:
            passes.append(_make_pass(row, sat, topo, aos, tca, t))
            aos, tca = None, None
    if aos is not None:
        passes.append(_make_pass(row, sat, topo, aos, tca, t1))
    return passes


def _station_position(station: Station) -> GeographicPosition:
    return cast("GeographicPosition", wgs84.latlon(station.lat_deg, station.lon_deg, elevation_m=station.alt_m))


def _elevation(sat: EarthSatellite, topo: GeographicPosition, t_unix: float) -> float:
    alt, _az, _d = (sat - topo).at(unix_to_time(t_unix)).altaz()
    return float(alt.degrees)


def _make_pass(
    row: dict[str, str], sat: EarthSatellite, topo: GeographicPosition, aos: float, tca: float | None, los: float
) -> Pass:
    if tca is None:  # clipped pass: culmination is at an end of the window, take the higher end
        tca = aos if _elevation(sat, topo, aos) >= _elevation(sat, topo, los) else los
    return Pass(int(row["NORAD_CAT_ID"]), row["OBJECT_NAME"], aos, tca, los, _elevation(sat, topo, tca))


@dataclass(frozen=True)
class Geometry:
    el_deg: np.ndarray
    az_deg: np.ndarray
    range_m: np.ndarray
    range_rate_m_s: np.ndarray

    def doppler_hz(self, f_hz: float) -> np.ndarray:
        """Received minus transmitted frequency for a carrier at f_hz (first order)."""
        return -f_hz * self.range_rate_m_s / C_M_S


def geometry(row: dict[str, str], station: Station, t_unix: np.ndarray) -> Geometry:
    """Topocentric geometry of a satellite at Unix times."""
    sat = satellite(row)
    topo = _station_position(station)
    pos = (sat - topo).at(unix_to_time(t_unix))
    alt, az, _dist = pos.altaz()
    _lat, _lon, dist, _lat_rate, _lon_rate, range_rate = pos.frame_latlon_and_rates(topo)
    return Geometry(
        np.asarray(alt.degrees),
        np.asarray(az.degrees),
        np.asarray(dist.m),
        np.asarray(range_rate.km_per_s) * 1e3,
    )
