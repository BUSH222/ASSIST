"""`assist catalog` (plan §6.1, §8): the element sets ADRIFT is given.

By default the whole element store (every object in `elements.files`, the set nearest the clock start), with the
scenario applied: leave-one-out sources (`in_catalog: false`) are removed, catalog sources keep their published set
(perturbed ones unperturbed, mimicking OMM error), and synthetic sources with `in_catalog: true` are added under
their catalog NORAD. `sources_only` restricts the output to the scenario's sources.
"""

import json
import math
from typing import TYPE_CHECKING, Any

from sgp4 import exporter, omm
from sgp4.api import Satrec

from assist.config import schema as S
from assist.orbits.elements import ElementStore
from assist.platforms.satellite import truth_satellite
from assist.timebase import parse_iso

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def _satrec(o: Mapping[str, Any]) -> Satrec:
    s = Satrec()
    omm.initialize(s, dict(o))
    return s


def synthetic_omm(sat: Any, name: str, norad: int) -> dict[str, Any]:  # sgp4 Satrec (C or Python)
    """OMM dict of a synthetic orbit (no international designator, so sgp4's export_omm cannot be used)."""
    return {
        "OBJECT_NAME": name,
        "OBJECT_ID": "",
        "EPOCH": exporter.sat_epoch_datetime(sat).strftime("%Y-%m-%dT%H:%M:%S.%f"),
        "MEAN_MOTION": sat.no_kozai * 1440.0 / (2 * math.pi),
        "ECCENTRICITY": sat.ecco,
        "INCLINATION": math.degrees(sat.inclo),
        "RA_OF_ASC_NODE": math.degrees(sat.nodeo),
        "ARG_OF_PERICENTER": math.degrees(sat.argpo),
        "MEAN_ANOMALY": math.degrees(sat.mo),
        "EPHEMERIS_TYPE": 0,
        "CLASSIFICATION_TYPE": "U",
        "NORAD_CAT_ID": norad,
        "ELEMENT_SET_NO": 999,
        "REV_AT_EPOCH": 0,
        "BSTAR": sat.bstar,
        "MEAN_MOTION_DOT": 0.0,
        "MEAN_MOTION_DDOT": 0.0,
    }


def entries(sc: S.Scenario, base: Path, sources_only: bool = False) -> list[dict[str, Any]]:
    """OMM dicts, sorted by NORAD id."""
    out: dict[int, dict[str, Any]] = {}
    if not sources_only and sc.elements.files:
        store = ElementStore.load([base / f for f in sc.elements.files])
        t0 = parse_iso(sc.clock.start)
        for n in store.norads():
            out[n] = dict(store.nearest(n, t0).omm)
    for src in sc.sources:
        o = src.orbit
        if o is None:  # ground and local sources are not in the catalog
            continue
        if isinstance(o, S.CatalogOrbit):
            if src.in_catalog and o.omm is not None:
                out[o.norad] = dict(o.omm)
            else:
                out.pop(o.norad, None)
        elif src.in_catalog and src.catalog_norad is not None:
            sat, _norad, name, _ = truth_satellite(src)
            out[src.catalog_norad] = synthetic_omm(sat.model, name, src.catalog_norad)
    return [out[k] for k in sorted(out)]


def write(entries_: list[dict[str, Any]], path: Path) -> None:
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(entries_, indent=1) + "\n")
        return
    lines = []
    for e in entries_:
        l1, l2 = exporter.export_tle(_satrec(e))
        lines += [str(e.get("OBJECT_NAME", "")) or str(e["NORAD_CAT_ID"]), l1, l2]
    path.write_text("\n".join(lines) + "\n")
