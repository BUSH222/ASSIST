"""Satellite platform: a truth orbit (catalog, Kepler, perturbed catalog or synthetic pass) and its spline geometry."""

from typing import TYPE_CHECKING, Any, override

from assist.config import schema as S
from assist.orbits.elements import satellite_from_omm
from assist.orbits.geometry import SplineGeometry
from assist.orbits.perturb import perturb_omm
from assist.orbits.synthetic import RE_KM, Kepler, kepler_satellite
from assist.platforms.base import Platform
from assist.timebase import parse_iso

if TYPE_CHECKING:
    import numpy as np
    from skyfield.api import EarthSatellite

    from assist.orbits.geometry import Geometry, Pass
    from assist.station import Station


def truth_satellite(src: S.Source) -> tuple[EarthSatellite, int | None, str, dict[str, Any] | None]:
    """(truth satellite, catalog NORAD or None, name, published OMM or None) of a resolved source."""
    o = src.orbit
    name = src.name or src.id
    if isinstance(o, S.CatalogOrbit):
        if o.omm is None:
            raise ValueError(f"source {src.id}: orbit not resolved (no element set embedded)")
        omm = o.omm
        if o.perturb is not None:
            p = o.perturb
            omm = perturb_omm(omm, p.along_track_s, p.raan_deg, p.inc_deg, p.dn_rev_day)
        return satellite_from_omm(omm), o.norad, src.name or str(o.omm.get("OBJECT_NAME", src.id)), o.omm
    satnum = src.catalog_norad or 99999
    if isinstance(o, S.KeplerOrbit):
        k = o.kepler
        a_km = k.a_km if k.a_km is not None else RE_KM + float(k.alt_km or 0.0)
        kep = Kepler(parse_iso(k.epoch), a_km, k.e, k.i_deg, k.raan_deg, k.argp_deg, k.m_deg, k.bstar)
    else:
        if o is None:
            raise ValueError(f"source {src.id}: a satellite needs an orbit")
        sp = o.synthetic_pass
        if sp.solution is None:
            raise ValueError(f"source {src.id}: synthetic pass not resolved")
        kep = Kepler(
            parse_iso(sp.aos), RE_KM + sp.alt_km, 0.0, sp.inc_deg, sp.solution.raan_deg, 0.0, sp.solution.m_deg
        )
    return kepler_satellite(kep, satnum, name), src.catalog_norad if src.in_catalog else None, name, None


class SatellitePlatform(Platform):
    def __init__(self, src: S.Source, station: Station, horizon_deg: float) -> None:
        self.source_id = src.id
        self.sat, self.norad, self.name, self.published_omm = truth_satellite(src)
        self.in_catalog = bool(src.in_catalog)
        self.horizon_deg = horizon_deg
        self.geo = SplineGeometry(self.sat, station, horizon_deg, self.norad or 0, self.name)

    @override
    def geometry(self, t_unix: np.ndarray) -> Geometry:
        return self.geo.geometry(t_unix)

    @override
    def passes(self, t0: float, t1: float) -> list[Pass]:
        return self.geo.passes(t0, t1)
