"""Element store: OMM CSV (retlector layout), OMM JSON and TLE files, and `assist elements fetch`.

Every element set of every file is kept per NORAD id. A scenario takes the set whose epoch is nearest its clock
start (ties go to the newer one); the profile tool keeps its own rule, the newest set per object (`load_newest`).
Files are only read here; the network is touched only by `fetch`, which the user runs explicitly.
"""

import csv
import datetime as dt
import json
import re
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from sgp4 import exporter
from sgp4.api import Satrec
from skyfield.api import EarthSatellite

from assist.timebase import timescale

if TYPE_CHECKING:
    from typing import BinaryIO

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
# Fields Skyfield/sgp4 need to build a satellite from an OMM dict.
OMM_REQUIRED = (
    "EPOCH",
    "MEAN_MOTION",
    "ECCENTRICITY",
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "MEAN_ANOMALY",
    "NORAD_CAT_ID",
    "BSTAR",
    "MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT",
)
FEEDS = {"active": "https://retlector.eu/active/csv", "satnogs": "https://retlector.eu/satnogs/csv"}
STALE_DAYS = 14.0

Omm = dict[str, Any]


def read_omm_csv(path: str | Path) -> list[dict[str, str]]:
    """Rows of an OMM CSV (retlector layout), checked by header name so upstream column changes fail loudly."""
    with Path(path).open(newline="") as f:
        return _omm_rows(csv.DictReader(f), str(path))


def _omm_rows(reader: csv.DictReader[str], what: str) -> list[dict[str, str]]:
    missing = set(OMM_COLUMNS) - set(reader.fieldnames or ())
    if missing:
        raise ValueError(f"{what}: missing OMM columns {sorted(missing)}")
    return list(reader)


def read_omm_json(path: str | Path) -> list[Omm]:
    """OMM JSON: a list of objects (or a single object) with the OMM keys."""
    d = json.loads(Path(path).read_text())
    rows = d if isinstance(d, list) else [d]
    for r in rows:
        missing = set(OMM_REQUIRED) - set(r)
        if missing:
            raise ValueError(f"{path}: OMM entry missing {sorted(missing)}")
    return rows


def read_tle(path: str | Path) -> list[Omm]:
    """Two-line or three-line element sets, converted to OMM dicts (sgp4's exporter)."""
    lines = [ln.rstrip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    out: list[Omm] = []
    i = 0
    while i < len(lines):
        name = ""
        if not lines[i].startswith("1 "):
            name = lines[i].removeprefix("0 ").strip()
            i += 1
        if i + 1 >= len(lines) or not (lines[i].startswith("1 ") and lines[i + 1].startswith("2 ")):
            raise ValueError(f"{path}: malformed TLE near line {i + 1}")
        sat = Satrec.twoline2rv(lines[i], lines[i + 1])
        out.append(exporter.export_omm(sat, name or str(sat.satnum)))
        i += 2
    return out


def read_elements(path: str | Path) -> list[Omm]:
    """Any supported element file, by extension: .csv (OMM), .json (OMM), .tle/.txt/.3le (TLE)."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return cast("list[Omm]", read_omm_csv(path))
    if suffix == ".json":
        return read_omm_json(path)
    if suffix in (".tle", ".txt", ".3le"):
        return read_tle(path)
    raise ValueError(f"{path}: unknown element format (use .csv, .json or .tle)")


def epoch_unix(omm: Mapping[str, Any]) -> float:
    """OMM EPOCH (UTC, usually without a zone) as Unix seconds."""
    d = dt.datetime.fromisoformat(str(omm["EPOCH"]).strip().removesuffix("Z"))
    d = d.replace(tzinfo=dt.UTC) if d.tzinfo is None else d
    return d.timestamp()


def satellite_from_omm(omm: Mapping[str, Any]) -> EarthSatellite:
    return cast("EarthSatellite", EarthSatellite.from_omm(timescale(), dict(omm)))


def load_newest(paths: Iterable[str | Path]) -> dict[int, dict[str, str]]:
    """Union of OMM CSV files by NORAD id; where several hold an object, the newest EPOCH wins (profile tool)."""
    out: dict[int, dict[str, str]] = {}
    for p in paths:
        for row in read_omm_csv(p):
            norad = int(row["NORAD_CAT_ID"])
            if norad not in out or row["EPOCH"] > out[norad]["EPOCH"]:  # ISO strings sort chronologically
                out[norad] = row
    return out


@dataclass(frozen=True)
class ElementSet:
    norad: int
    name: str
    epoch_unix: float
    omm: Omm = field(compare=False)
    source: str = field(default="", compare=False)


class ElementStore:
    """All element sets from a list of files, per NORAD id, sorted by epoch."""

    def __init__(self, sets: Iterable[ElementSet] = ()) -> None:
        self._by_norad: dict[int, list[ElementSet]] = {}
        for s in sets:
            lst = self._by_norad.setdefault(s.norad, [])
            if all(x.epoch_unix != s.epoch_unix for x in lst):  # same object, same epoch in two feeds: keep first
                lst.append(s)
        for lst in self._by_norad.values():
            lst.sort(key=lambda s: s.epoch_unix)

    @classmethod
    def load(cls, paths: Iterable[str | Path]) -> ElementStore:
        sets: list[ElementSet] = []
        for p in paths:
            for omm in read_elements(p):
                sets.append(
                    ElementSet(int(omm["NORAD_CAT_ID"]), str(omm.get("OBJECT_NAME", "")), epoch_unix(omm), omm, str(p))
                )
        return cls(sets)

    def norads(self) -> list[int]:
        return sorted(self._by_norad)

    def sets(self, norad: int) -> list[ElementSet]:
        return list(self._by_norad.get(norad, []))

    def nearest(self, norad: int, t_unix: float) -> ElementSet:
        """The set whose epoch is nearest t (ties: the newer one)."""
        lst = self._by_norad.get(norad)
        if not lst:
            raise KeyError(f"NORAD {norad} is in none of the element files")
        return min(lst, key=lambda s: (abs(s.epoch_unix - t_unix), -s.epoch_unix))


def staleness_warning(es: ElementSet, t_unix: float) -> str | None:
    """A warning when the window starts more than 14 days after the set's epoch.

    Epochs after the window never warn: predictive element sets (e.g. CXO, XMM-Newton) have future epochs on
    purpose, and back-propagation to past dates is the documented way to cover them.
    """
    age_days = (t_unix - es.epoch_unix) / 86400.0
    if age_days > STALE_DAYS:
        return f"NORAD {es.norad} ({es.name}): elements are {age_days:.1f} days older than the window"
    return None


def snapshot_name(feed: str, now: dt.datetime) -> str:
    return f"{feed}_{now.astimezone(dt.UTC):%Y-%m-%d}.csv"


Opener = Callable[[str], "BinaryIO"]
ELEMENTS_DIR = "elements"
_SNAPSHOT = re.compile(r"^(?P<feed>[a-z]+)_(?P<date>\d{4}-\d{2}-\d{2})\.csv$")


def _open_url(url: str) -> BinaryIO:
    req = urllib.request.Request(url, headers={"User-Agent": "assist-elements-fetch"})
    return cast("BinaryIO", urllib.request.urlopen(req, timeout=60))  # fixed https URLs only


def fetch(
    feeds: Iterable[str],
    out_dir: str | Path,
    now: dt.datetime | None = None,
    opener: Opener = _open_url,
) -> list[Path]:
    """Download retlector feeds into `<out_dir>/<feed>_<UTC date>.csv`. Never overwrites an existing file.

    Each download is checked (OMM header, parseable rows) in a `.part` file before it is renamed into place.
    """
    now = now or dt.datetime.now(dt.UTC)
    out = Path(out_dir)
    targets = []
    for feed in feeds:
        if feed not in FEEDS:
            raise ValueError(f"unknown feed {feed!r} (known: {', '.join(FEEDS)})")
        target = out / snapshot_name(feed, now)
        if target.exists():
            raise FileExistsError(f"{target} exists; snapshots are never overwritten")
        targets.append((feed, target))
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for feed, target in targets:
        part = target.with_suffix(".csv.part")
        with opener(FEEDS[feed]) as r:
            data = r.read()
        text = data.decode("utf-8")
        rows = _omm_rows(csv.DictReader(text.splitlines()), FEEDS[feed])
        if not rows:
            raise ValueError(f"{FEEDS[feed]}: no element sets")
        for row in rows:
            int(row["NORAD_CAT_ID"])
            epoch_unix(row)
        part.write_bytes(data)
        part.rename(target)
        written.append(target)
    return written


def existing_snapshots(out_dir: str | Path) -> dict[str, Path]:
    """The newest `<feed>_<date>.csv` per known feed in `out_dir` (feeds without a snapshot are absent)."""
    best: dict[str, tuple[str, Path]] = {}
    d = Path(out_dir)
    for p in d.iterdir() if d.is_dir() else ():
        m = _SNAPSHOT.match(p.name)
        if m and m["feed"] in FEEDS and (m["feed"] not in best or m["date"] > best[m["feed"]][0]):
            best[m["feed"]] = (m["date"], p)
    return {feed: best[feed][1] for feed in FEEDS if feed in best}


def latest_snapshots(
    out_dir: str | Path,
    download: bool,
    now: dt.datetime | None = None,
    opener: Opener | None = None,
) -> tuple[list[Path], list[str]]:
    """The element files a scenario's `latest` / `latest_existing` stands for, and warnings.

    download=False (`latest_existing`): the newest snapshot per feed already in `out_dir`.
    download=True (`latest`): today's (UTC) snapshot per feed; the missing ones are fetched first. If a fetch fails
    the newest existing snapshot of that feed is used, with a warning. Raises FileNotFoundError when no file results.
    """
    now = now or dt.datetime.now(dt.UTC)
    out = Path(out_dir)
    warnings: list[str] = []
    if download:
        missing = [f for f in FEEDS if not (out / snapshot_name(f, now)).exists()]
        if missing:
            try:
                fetch(missing, out, now, opener or _open_url)
            except (OSError, ValueError) as e:  # URLError is an OSError
                warnings.append(f"could not download today's element sets ({e}); using the newest files in {out}")
    found = existing_snapshots(out)
    if not found:
        hint = "" if download else " (run `assist fetch_omm`)"
        raise FileNotFoundError(f"no element snapshots (<feed>_<date>.csv) in {out}{hint}")
    return list(found.values()), warnings
