"""Synthetic orbits (plan §6.1): Kepler elements as SGP4 mean elements, and passes solved for AOS and max elevation.

A synthetic pass is a circular orbit (e = 0, argument of perigee 0) with the requested altitude and inclination,
epoch at the requested AOS. RAAN and mean anomaly are seeded from spherical geometry, then refined with
scipy.optimize.least_squares on the AOS-time and max-elevation errors measured with Skyfield.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from scipy.optimize import brentq, least_squares, minimize_scalar
from sgp4.api import WGS72, Satrec
from skyfield.api import EarthSatellite
from skyfield.sgp4lib import theta_GMST1982

from assist.orbits.geometry import station_position, topocentric
from assist.timebase import format_iso, timescale, unix_to_time, unix_to_time_fast

if TYPE_CHECKING:
    from collections.abc import Callable

    from assist.station import Station

MU_KM3_S2 = 398600.8  # WGS72, the gravity model SGP4 uses
RE_KM = 6378.135  # WGS72 equatorial radius
EPOCH_1949_UNIX = -631238400.0  # 1949-12-31 00:00 UTC, SGP4's epoch origin
OMEGA_EARTH = 7.292115e-5  # rad/s


@dataclass(frozen=True)
class Kepler:
    epoch_unix: float
    a_km: float
    e: float = 0.0
    i_deg: float = 0.0
    raan_deg: float = 0.0
    argp_deg: float = 0.0
    m_deg: float = 0.0
    bstar: float = 0.0


def kepler_satrec(k: Kepler, satnum: int = 99999) -> Satrec:
    """Satrec.sgp4init from mean elements; mean motion from the semi-major axis (Kepler's third law)."""
    n_rad_min = math.sqrt(MU_KM3_S2 / k.a_km**3) * 60.0
    s = Satrec()
    s.sgp4init(
        WGS72,
        "i",
        satnum,
        (k.epoch_unix - EPOCH_1949_UNIX) / 86400.0,
        k.bstar,
        0.0,
        0.0,
        k.e,
        math.radians(k.argp_deg),
        math.radians(k.i_deg),
        math.radians(k.m_deg),
        n_rad_min,
        math.radians(k.raan_deg),
    )
    return s


def kepler_satellite(k: Kepler, satnum: int = 99999, name: str = "") -> EarthSatellite:
    sat = cast("EarthSatellite", EarthSatellite.from_satrec(kepler_satrec(k, satnum), timescale()))
    sat.name = name or None
    return sat


@dataclass(frozen=True)
class PassRequest:
    aos_unix: float
    max_el_deg: float
    direction: Literal["asc", "desc"]
    side: Literal["east", "west"]
    alt_km: float = 820.0
    inc_deg: float = 98.7


def _station_teme_km(station: Station, t_unix: float) -> np.ndarray:
    """Station position in TEME (km): ITRS rotated by GMST (polar motion ignored; seed geometry only)."""
    x, y, z = station_position(station).itrs_xyz.km
    jd_ut1 = unix_to_time(t_unix).ut1
    theta, _ = theta_GMST1982(jd_ut1)
    c, s = math.cos(theta), math.sin(theta)
    return np.array([c * x - s * y, s * x + c * y, z])


def _seeds(req: PassRequest, station: Station, horizon_deg: float) -> list[tuple[float, float, float, str, str]]:
    """(raan_deg, m_deg, cross-track sign, direction, side) candidates.

    Spherical geometry at epoch, with the station rotated to the expected culmination time.
    """
    r_km = RE_KM + req.alt_km
    inc = math.radians(req.inc_deg)
    el_max, el_h = math.radians(min(req.max_el_deg, 89.999)), math.radians(horizon_deg)
    beta_c = math.acos(RE_KM * math.cos(el_max) / r_km) - el_max  # central angle at culmination
    beta_h = math.acos(RE_KM * math.cos(el_h) / r_km) - el_h  # central angle at AOS
    alpha = math.acos(min(1.0, math.cos(beta_h) / math.cos(beta_c)))  # along-track angle AOS -> culmination
    n = math.sqrt(MU_KM3_S2 / r_km**3)
    s_vec = _station_teme_km(station, req.aos_unix + alpha / n)  # station near culmination time
    s_hat = s_vec / np.linalg.norm(s_vec)
    rho, lam = math.hypot(s_hat[0], s_hat[1]), math.atan2(s_hat[1], s_hat[0])
    out = []
    for sigma in (1.0, -1.0):
        q = (math.sin(sigma * beta_c) - math.cos(inc) * s_hat[2]) / (rho * math.sin(inc))
        if abs(q) > 1:
            continue
        for raan in (lam + math.asin(q), lam + math.pi - math.asin(q)):
            n_hat = np.array([math.sin(inc) * math.sin(raan), -math.sin(inc) * math.cos(raan), math.cos(inc)])
            node = np.array([math.cos(raan), math.sin(raan), 0.0])
            w_hat = np.cross(n_hat, node)
            c = s_hat - np.dot(n_hat, s_hat) * n_hat
            u_c = math.atan2(float(np.dot(c, w_hat)), float(np.dot(c, node)))
            direction = "asc" if math.cos(u_c) > 0 else "desc"
            sat_c = r_km * (math.cos(u_c) * node + math.sin(u_c) * w_hat)
            d = sat_c - s_vec
            east = np.cross([0.0, 0.0, 1.0], s_hat)
            east /= np.linalg.norm(east)
            north = np.cross(s_hat, east)
            az = math.degrees(math.atan2(float(np.dot(d, east)), float(np.dot(d, north)))) % 360
            side = "east" if 0 < az < 180 else "west"
            m = math.degrees(u_c - alpha) % 360
            out.append((math.degrees(raan) % 360, m, sigma, direction, side))
    return out


def _el_fn(sat: EarthSatellite, station: Station) -> Callable[[np.ndarray | float], np.ndarray]:
    def el(t: np.ndarray | float) -> np.ndarray:
        return topocentric(sat, station, np.atleast_1d(np.asarray(t, dtype=np.float64)), fast=True).el_deg

    return el


def measure_pass(
    sat: EarthSatellite, station: Station, horizon_deg: float, near_unix: float
) -> tuple[float, float, float]:
    """(AOS, max elevation, cross-track sign) of the pass rising nearest `near_unix` (direct Skyfield).

    The sign is +1 when the station lies on the orbit-normal side of the track at culmination (r x v points
    towards it), -1 otherwise.
    """
    el = _el_fn(sat, station)
    grid = near_unix + np.arange(-1800.0, 1801.0, 5.0)
    e = el(grid) - horizon_deg
    rising = np.flatnonzero((e[:-1] < 0) & (e[1:] >= 0))
    if len(rising) == 0:
        raise ValueError("no rising horizon crossing within 30 min")
    i = int(rising[np.argmin(np.abs(grid[rising] - near_unix))])
    a0 = float(grid[i])
    aos = a0 + float(brentq(lambda x: float(el(a0 + x)[0]) - horizon_deg, 0.0, 5.0, xtol=1e-7))
    setting = np.flatnonzero(e[i + 1 :] < 0)
    k = i + 1 + (int(setting[0]) if len(setting) else len(e) - i - 1)  # first sample below the horizon
    j = i + 1 + int(np.argmax(e[i + 1 : k]))
    lo, hi = float(grid[max(j - 1, 0)]), float(grid[min(j + 1, len(grid) - 1)])
    # Relative time: bounded Brent's tolerance is relative to |x| (27 s at Unix 1.8e9 s).
    r = minimize_scalar(
        lambda x: -float(el(lo + x)[0]), bounds=(0.0, hi - lo), method="bounded", options={"xatol": 1e-4}
    )
    tt = unix_to_time_fast(lo + float(r.x))
    p = sat.at(tt)
    normal = np.cross(np.asarray(p.position.m), np.asarray(p.velocity.m_per_s))
    side = float(np.dot(normal, np.asarray(station_position(station).at(tt).position.m)))
    return aos, -float(r.fun), 1.0 if side > 0 else -1.0


def solve_pass(req: PassRequest, station: Station, horizon_deg: float = 0.0) -> tuple[float, float]:
    """(raan_deg, m_deg) of the circular orbit whose pass rises at req.aos_unix and culminates at req.max_el_deg."""
    if not horizon_deg < req.max_el_deg <= 90:
        raise ValueError(f"max_el_deg must be in ({horizon_deg}, 90], got {req.max_el_deg}")
    seeds = [s for s in _seeds(req, station, horizon_deg) if s[3] == req.direction and s[4] == req.side]
    if not seeds:
        raise ValueError(
            f"no {req.direction} pass passing {req.side} at {req.max_el_deg} deg for inclination {req.inc_deg} deg"
            " from this station"
        )
    raan0, m0, sigma, _, _ = seeds[0]
    # Zenith angle at culmination signed by the side of the track: smooth through the zenith, unlike max elevation.
    target_z = (90.0 - req.max_el_deg) * sigma

    def sat_for(x: np.ndarray) -> EarthSatellite:
        k = Kepler(req.aos_unix, RE_KM + req.alt_km, 0.0, req.inc_deg, float(x[0]) % 360, 0.0, float(x[1]) % 360)
        return kepler_satellite(k)

    def resid(x: np.ndarray) -> np.ndarray:
        try:
            aos, mx, sign = measure_pass(sat_for(x), station, horizon_deg, req.aos_unix)
        except ValueError:
            return np.array([1e4, 1e4])
        return np.array([(aos - req.aos_unix) / 0.1, ((90.0 - mx) * sign - target_z) / 0.005])

    r = least_squares(resid, np.array([raan0, m0]), x_scale=1e-2, xtol=1e-10, ftol=1e-10, max_nfev=100)
    raan, m = float(r.x[0]) % 360, float(r.x[1]) % 360
    aos, mx, _ = measure_pass(sat_for(r.x), station, horizon_deg, req.aos_unix)
    if abs(aos - req.aos_unix) > 0.5 or abs(mx - req.max_el_deg) > 0.1:
        raise ValueError(
            f"synthetic pass did not converge: AOS {aos - req.aos_unix:+.2f} s, max el {mx:.2f} deg"
            f" (wanted {format_iso(req.aos_unix)}, {req.max_el_deg} deg)"
        )
    return raan, m
