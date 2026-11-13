"""Satellite platform: a truth orbit (catalog, Kepler, perturbed catalog or synthetic pass) and its spline geometry."""

import math
from typing import TYPE_CHECKING, Any, override

from assist.config import schema as S
from assist.orbits.elements import satellite_from_omm
from assist.orbits.geometry import SplineGeometry
from assist.orbits.perturb import perturb_omm
from assist.orbits.synthetic import RE_KM, Kepler, kepler_satellite
from assist.platforms.base import Platform, PlatformContext, register_platform
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


MU_KM3_S2 = 398600.8  # WGS72, as SGP4
EARTH_ROTATION_KM_S = 0.6  # bound on what Earth rotation adds to a LEO satellite's speed seen from the ground


def max_speed_km_s(sat: object) -> float:
    """Upper bound on a satellite's speed relative to the station: perigee speed plus Earth rotation."""
    model = sat.model  # ty: ignore[unresolved-attribute]
    n = float(model.no_kozai) / 60.0  # rad/s
    e = float(model.ecco)
    a = (MU_KM3_S2 / n**2) ** (1 / 3)
    rp = a * (1 - e)
    return math.sqrt(MU_KM3_S2 * (2 / rp - 1 / a)) + EARTH_ROTATION_KM_S


@register_platform("satellite")
class SatellitePlatform(Platform):
    needs_orbit = True
    catalog = True
    supports_oscillator = True
    level_mode = "link_budget"
    fades_at_horizon = True

    def __init__(self, src: S.Source, station: Station, horizon_deg: float) -> None:
        self.source_id = src.id
        self.sat, self.norad, self.name, self.published_omm = truth_satellite(src)
        self.in_catalog = bool(src.in_catalog)
        self.horizon_deg = horizon_deg
        self.geo = SplineGeometry(self.sat, station, horizon_deg, self.norad or 0, self.name)

    @override
    @classmethod
    def build(cls, ctx: PlatformContext) -> SatellitePlatform:
        return cls(ctx.source, ctx.station, ctx.horizon_deg)

    @override
    def geometry(self, t_unix: np.ndarray) -> Geometry:
        return self.geo.geometry(t_unix)

    @override
    def geometry_ms(self, k_ms: np.ndarray) -> Geometry:
        return self.geo.geometry_ms(k_ms)

    @override
    def pass_keys(self, t_unix: np.ndarray, passes: list[Pass]) -> np.ndarray:
        return self.geo.pass_keys(t_unix, passes)

    @override
    def max_speed_km_s(self) -> float:
        return max_speed_km_s(self.sat)

    @override
    def passes(self, t0: float, t1: float) -> list[Pass]:
        return self.geo.passes(t0, t1)
