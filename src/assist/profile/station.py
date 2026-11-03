"""Station position for the profile tool. Moves into the scenario config in M2."""

from dataclasses import dataclass
from pathlib import Path

import yaml

# Plan §0 / Appendix A defaults; samples/station.yaml overrides lat/lon (and alt/tz if given).
DEFAULT_ALT_M = 180.0
DEFAULT_TZ = "+03:00"


@dataclass(frozen=True)
class Station:
    lat_deg: float
    lon_deg: float
    alt_m: float = DEFAULT_ALT_M
    tz: str = DEFAULT_TZ  # local time of SDR++ file names


def load_station(path: str | Path) -> Station:
    """Read a station file: `station_lat`, `station_lon` (or `lat`, `lon`), optional altitude and time zone.

    Altitude keys: `altitude_m`, `alt_m`, `station_alt_m`; time-zone keys: `tz`, `station_tz`.
    """
    d = yaml.safe_load(Path(path).read_text()) or {}

    def get(*names: str) -> object:
        for n in names:
            if n in d:
                return d[n]
        return None

    lat, lon = get("station_lat", "lat"), get("station_lon", "lon")
    if lat is None or lon is None:
        raise ValueError(f"{path}: needs station_lat and station_lon")
    alt = get("altitude_m", "alt_m", "station_alt_m")
    tz = get("station_tz", "tz")
    return Station(
        float(lat),  # ty: ignore[invalid-argument-type]
        float(lon),  # ty: ignore[invalid-argument-type]
        DEFAULT_ALT_M if alt is None else float(alt),  # ty: ignore[invalid-argument-type]
        DEFAULT_TZ if tz is None else str(tz),
    )
