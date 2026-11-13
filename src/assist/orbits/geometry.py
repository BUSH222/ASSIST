"""Satellite geometry seen from the station.

Three layers:
- Direct Skyfield evaluation (`topocentric`, `geometry`, `find_passes`): what the profile tool uses, and the
  reference the tests hold the simulator to.
- Direct SGP4 evaluation (`sgp4_topocentric`): the satellite's SGP4 model in TEME, rotated to Earth-fixed axes by
  GMST 1982 at UT1 (as Skyfield's TEME frame does), seen from the station's WGS84 position. No precession or
  nutation is needed, since the station-to-satellite geometry does not depend on the inertial frame used. Polar
  motion is ignored, as Skyfield does without a loaded table. Against full Skyfield: range within ~1 mm (LEO) to
  ~4 cm (15 h deep-space passes), angles within 1e-6 deg, about 30x faster.
- `SplineGeometry` for the simulator: direct SGP4 sampled at whole Unix seconds over aligned one-hour blocks (plus
  2 min each side), turned into cubic splines. Range rate is the derivative of the range spline. Because the sample
  grid never depends on the requested window, any window gives the same values (random access). Passes come from
  the same samples, so a pass's key second (`aos_key`, the whole second nearest the AOS) is the same
  whichever window asked for it.
"""

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar
from skyfield.api import wgs84

from assist.orbits.elements import satellite_from_omm
from assist.timebase import unix_to_time, unix_to_time_fast

if TYPE_CHECKING:
    from collections.abc import Mapping

    from skyfield.api import EarthSatellite, Time
    from skyfield.toposlib import GeographicPosition

    from assist.station import Station

C_M_S = 299_792_458.0
DAY_S = 86400
JD_UNIX_EPOCH = 2440587.5  # Julian date of 1970-01-01 00:00 UTC
JD_J2000 = 2451545.0
OMEGA_EARTH = 7.292115146706979e-5  # rad/s, the Earth rotation rate with GMST 1982 (Vallado)
BLOCK_S = 3600
MARGIN_S = 120
LOOK_BLOCKS = 24  # how far (in blocks) to look for the AOS of a pass in progress, or the LOS of an open one


@dataclass(frozen=True)
class Pass:
    norad: int
    name: str
    aos: float  # Unix seconds (horizon crossing at min_el)
    tca: float  # culmination
    los: float
    max_el_deg: float
    aos_key: int | None = None  # simulator passes: the whole Unix second nearest the AOS (RNG key)


@dataclass(frozen=True)
class Geometry:
    el_deg: np.ndarray
    az_deg: np.ndarray
    range_m: np.ndarray
    range_rate_m_s: np.ndarray
    off_nadir_deg: np.ndarray  # angle at the satellite between (geocentric) nadir and the station

    def doppler_hz(self, f_hz: float | np.ndarray) -> np.ndarray:
        """Received minus transmitted frequency for a carrier at f_hz (first order)."""
        return -f_hz * self.range_rate_m_s / C_M_S


def station_position(station: Station) -> GeographicPosition:
    return cast("GeographicPosition", wgs84.latlon(station.lat_deg, station.lon_deg, elevation_m=station.alt_m))


def _evaluate(sat: EarthSatellite, topo: GeographicPosition, tt: Time) -> Geometry:
    pos = (sat - topo).at(tt)
    alt, az, _dist = pos.altaz()
    _lat, _lon, dist, _lat_rate, _lon_rate, range_rate = pos.frame_latlon_and_rates(topo)
    d = np.asarray(pos.position.m)  # station -> satellite
    r_sat = np.asarray(sat.at(tt).position.m)  # Earth centre -> satellite
    cross = np.linalg.norm(np.cross(r_sat, d, axis=0), axis=0)
    off_nadir = np.degrees(np.arctan2(cross, np.sum(r_sat * d, axis=0)))
    return Geometry(
        np.asarray(alt.degrees),
        np.asarray(az.degrees),
        np.asarray(dist.m),
        np.asarray(range_rate.km_per_s) * 1e3,
        off_nadir,
    )


def topocentric(sat: EarthSatellite, station: Station, t_unix: np.ndarray, fast: bool = False) -> Geometry:
    """Topocentric geometry straight from Skyfield (full IAU 2000A nutation unless fast)."""
    tt = (unix_to_time_fast if fast else unix_to_time)(np.asarray(t_unix, dtype=np.float64))
    return _evaluate(sat, station_position(station), tt)


def geometry(row: Mapping[str, Any], station: Station, t_unix: np.ndarray) -> Geometry:
    """Topocentric geometry of a satellite given by an OMM row (profile tool)."""
    return topocentric(satellite_from_omm(row), station, t_unix)


# --- simulator: direct SGP4 evaluation ---


@dataclass(frozen=True)
class StationFrame:
    """The station's Earth-fixed position (km) and local east, north, up unit vectors (WGS84 geodetic)."""

    ecef_km: np.ndarray
    east: np.ndarray
    north: np.ndarray
    up: np.ndarray


def station_frame(station: Station) -> StationFrame:
    lat, lon = np.radians(station.lat_deg), np.radians(station.lon_deg)
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    return StationFrame(
        np.asarray(station_position(station).itrs_xyz.km, dtype=np.float64),  # Skyfield's own WGS84 position
        np.array([-so, co, 0.0]),
        np.array([-sl * co, -sl * so, cl]),
        np.array([cl * co, cl * so, sl]),
    )


def gmst82(jd_whole: np.ndarray | float, jd_frac: np.ndarray | float) -> np.ndarray:
    """Greenwich mean sidereal time (rad), IAU 1982, of a UT1 Julian date given in two parts.

    The formula of sgp4's `gstime` (and Skyfield's TEME frame), vectorised. The parts are never added into one float:
    that alone would cost ~40 us of time resolution (~1 cm of station position).
    """
    tut1 = ((np.asarray(jd_whole) - JD_J2000) + np.asarray(jd_frac)) / 36525.0
    s = -6.2e-6 * tut1**3 + 0.093104 * tut1**2 + (876600.0 * 3600 + 8640184.812866) * tut1 + 67310.54841
    return np.mod(np.radians(s / 240.0), 2 * np.pi)


def sgp4_topocentric(model: Any, frame: StationFrame, t0: int, dt_s: np.ndarray) -> Geometry:  # model: sgp4 Satrec
    """Geometry at Unix times t0 + dt_s from the satellite's SGP4 model directly (no Skyfield in the loop).

    t0 is a whole Unix second and dt_s the offsets from it, so times keep full precision. UT1 - UTC comes from
    Skyfield's built-in table at every time, which keeps the result a pure function of time (one value per block would
    step the station by ~1 cm at block boundaries). Range rate here uses SGP4's velocity; SplineGeometry takes the
    derivative of its range spline instead. SGP4 errors (e.g. decayed orbits) give NaN, as with Skyfield.
    """
    dt_s = np.asarray(dt_s, dtype=np.float64)
    day0 = t0 // DAY_S
    sec = float(t0 - day0 * DAY_S) + dt_s  # seconds from the UTC midnight starting day0
    jd_whole = float(day0) + JD_UNIX_EPOCH
    _err, r, v = model.sgp4_array(np.full(dt_s.shape, jd_whole), sec / DAY_S)  # TEME, km, km/s; UTC as Skyfield
    dut1 = np.asarray(unix_to_time(t0 + dt_s).dut1, dtype=np.float64)
    th = gmst82(jd_whole, (sec + dut1) / DAY_S)
    c, s = np.cos(th), np.sin(th)
    x = c * r[..., 0] + s * r[..., 1]  # TEME -> Earth-fixed (pseudo-Earth-fixed; polar motion ignored)
    y = -s * r[..., 0] + c * r[..., 1]
    r_sat = np.stack([x, y, r[..., 2]], axis=-1)
    v_sat = np.stack(
        [c * v[..., 0] + s * v[..., 1] + OMEGA_EARTH * y, -s * v[..., 0] + c * v[..., 1] - OMEGA_EARTH * x, v[..., 2]],
        axis=-1,
    )
    d = r_sat - frame.ecef_km  # station -> satellite
    rng = np.linalg.norm(d, axis=-1)
    cross = np.linalg.norm(np.cross(r_sat, d), axis=-1)
    return Geometry(
        np.degrees(np.arcsin(np.clip((d @ frame.up) / rng, -1.0, 1.0))),
        np.degrees(np.arctan2(d @ frame.east, d @ frame.north)) % 360.0,
        rng * 1e3,
        np.sum(d * v_sat, axis=-1) / rng * 1e3,
        np.degrees(np.arctan2(cross, np.sum(r_sat * d, axis=-1))),
    )


# --- profile tool: passes by Skyfield's find_events, clipped to the window ---


def find_passes(row: Mapping[str, Any], station: Station, t0: float, t1: float, min_el_deg: float = 0.0) -> list[Pass]:
    """Passes above min_el overlapping [t0, t1]. Passes in progress at t0 or t1 are clipped to the window."""
    sat = satellite_from_omm(row)
    topo = station_position(station)
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


def _elevation(sat: EarthSatellite, topo: GeographicPosition, t_unix: float) -> float:
    alt, _az, _d = (sat - topo).at(unix_to_time(t_unix)).altaz()
    return float(alt.degrees)


def _make_pass(
    row: Mapping[str, Any], sat: EarthSatellite, topo: GeographicPosition, aos: float, tca: float | None, los: float
) -> Pass:
    if tca is None:  # clipped pass: culmination is at an end of the window, take the higher end
        tca = aos if _elevation(sat, topo, aos) >= _elevation(sat, topo, los) else los
    return Pass(int(row["NORAD_CAT_ID"]), str(row["OBJECT_NAME"]), aos, tca, los, _elevation(sat, topo, tca))


# --- simulator: spline geometry on aligned blocks ---


@dataclass(frozen=True)
class _Crossing:
    t: float  # refined crossing time
    rising: bool
    # The whole Unix second nearest the crossing. Nearest, not the first one after it: a synthetic pass is solved to
    # rise within nanoseconds of a requested whole second, which "first after" would turn into a coin toss between
    # two keys; "nearest" puts the flip 0.5 s away from it.
    key: int


class _Block:
    def __init__(self, model: Any, frame: StationFrame, index: int, horizon_deg: float) -> None:  # sgp4 Satrec
        self.t0 = index * BLOCK_S
        self.ts = np.arange(-MARGIN_S, BLOCK_S + MARGIN_S + 1, dtype=np.float64)  # seconds from t0
        g = sgp4_topocentric(model, frame, self.t0, self.ts)
        self.samples = g
        self.range = CubicSpline(self.ts, g.range_m)
        self.range_rate = self.range.derivative()
        self.el = CubicSpline(self.ts, g.el_deg)
        self.az = CubicSpline(self.ts, np.unwrap(np.radians(g.az_deg)))
        self.off_nadir = CubicSpline(self.ts, g.off_nadir_deg)
        self.crossings = self._crossings(horizon_deg)

    def _crossings(self, horizon_deg: float) -> list[_Crossing]:
        """Horizon crossings between whole seconds s and s+1, for s in this block's own hour."""
        own = slice(MARGIN_S, MARGIN_S + BLOCK_S + 1)  # samples t0 .. t0 + 3600
        vis = self.samples.el_deg[own] >= horizon_deg
        out = []
        for i in np.flatnonzero(vis[:-1] != vis[1:]):
            a = float(self.ts[MARGIN_S + i])
            t = brentq(lambda x: float(self.el(x)) - horizon_deg, a, a + 1.0, xtol=1e-9)
            out.append(_Crossing(self.t0 + t, bool(vis[i + 1]), self.t0 + math.floor(t + 0.5)))
        return out

    def geometry(self, t_rel: np.ndarray) -> Geometry:
        return Geometry(
            self.el(t_rel),
            np.degrees(np.mod(self.az(t_rel), 2 * np.pi)),
            self.range(t_rel),
            self.range_rate(t_rel),
            self.off_nadir(t_rel),
        )


class SplineGeometry:
    """Spline-sampled geometry of one satellite from one station, with passes."""

    def __init__(
        self,
        sat: EarthSatellite,
        station: Station,
        horizon_deg: float = 0.0,
        norad: int = 0,
        name: str = "",
        cache_blocks: int = 8,
    ) -> None:
        self.sat, self.station, self.horizon_deg = sat, station, horizon_deg
        self.norad, self.name = norad, name
        self._model = sat.model
        self._frame = station_frame(station)
        self._cache: OrderedDict[int, _Block] = OrderedDict()
        self._cache_blocks = cache_blocks

    def _block(self, index: int) -> _Block:
        b = self._cache.get(index)
        if b is None:
            b = _Block(self._model, self._frame, index, self.horizon_deg)
            self._cache[index] = b
            while len(self._cache) > self._cache_blocks:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(index)
        return b

    def geometry(self, t_unix: np.ndarray | float) -> Geometry:
        t = np.atleast_1d(np.asarray(t_unix, dtype=np.float64))
        idx = np.floor(t / BLOCK_S).astype(np.int64)
        out = {k: np.empty(t.shape) for k in ("el", "az", "r", "rr", "on")}
        for b in np.unique(idx):
            m = idx == b
            blk = self._block(int(b))
            g = blk.geometry(t[m] - blk.t0)
            out["el"][m], out["az"][m], out["r"][m] = g.el_deg, g.az_deg, g.range_m
            out["rr"][m], out["on"][m] = g.range_rate_m_s, g.off_nadir_deg
        return Geometry(out["el"], out["az"], out["r"], out["rr"], out["on"])

    def geometry_ms(self, k_ms: np.ndarray) -> Geometry:
        """Geometry at integer Unix milliseconds (the renderer's control grid).

        Float Unix seconds resolve only ~0.24 us (range error up to ~2 mm, i.e. phase jitter of a few mrad at
        137 MHz). Here the time within the block is formed from integers first, so it is exact to ~1e-13 s.
        """
        k = np.asarray(k_ms, dtype=np.int64)
        idx = k // (BLOCK_S * 1000)
        out = {n: np.empty(k.shape) for n in ("el", "az", "r", "rr", "on")}
        for b in np.unique(idx):
            m = idx == b
            blk = self._block(int(b))
            g = blk.geometry((k[m] - int(b) * BLOCK_S * 1000) / 1000.0)
            out["el"][m], out["az"][m], out["r"][m] = g.el_deg, g.az_deg, g.range_m
            out["rr"][m], out["on"][m] = g.range_rate_m_s, g.off_nadir_deg
        return Geometry(out["el"], out["az"], out["r"], out["rr"], out["on"])

    def _crossings_between(self, a: float, b: float) -> list[_Crossing]:
        out: list[_Crossing] = []
        for i in range(int(np.floor(a / BLOCK_S)), int(np.floor(b / BLOCK_S)) + 1):
            out += [c for c in self._block(i).crossings if a <= c.t < b]
        return out

    def _last_before(self, t: float) -> _Crossing | None:
        i0 = int(np.floor(t / BLOCK_S))
        for i in range(i0, i0 - LOOK_BLOCKS - 1, -1):
            cs = [c for c in self._block(i).crossings if c.t < t]
            if cs:
                return cs[-1]
        return None

    def _first_after(self, t: float) -> _Crossing | None:
        i0 = int(np.floor(t / BLOCK_S))
        for i in range(i0, i0 + LOOK_BLOCKS + 1):
            cs = [c for c in self._block(i).crossings if c.t >= t]
            if cs:
                return cs[0]
        return None

    def passes(self, t0: float, t1: float) -> list[Pass]:
        """Complete passes (true AOS and LOS, not clipped) that overlap [t0, t1)."""
        cs = self._crossings_between(t0, t1)
        prev = self._last_before(t0)
        if prev is None and float(self.geometry(t0).el_deg[0]) >= self.horizon_deg:
            raise ValueError(f"{self.name or self.norad}: above the horizon for > {LOOK_BLOCKS} h; no pass bounds")
        if prev is not None and prev.rising:
            cs = [prev, *cs]
        if cs and cs[-1].rising:
            nxt = self._first_after(cs[-1].t + 1e-6)
            if nxt is None:
                raise ValueError(f"{self.name or self.norad}: no LOS within {LOOK_BLOCKS} h of {cs[-1].t}")
            cs.append(nxt)
        out = []
        for rise, sett in zip(cs[0::2], cs[1::2], strict=True):
            if not rise.rising or sett.rising:
                raise RuntimeError("horizon crossings out of order")  # cannot happen with consistent samples
            tca, el = self._culmination(rise.t, sett.t)
            out.append(Pass(self.norad, self.name, rise.t, tca, sett.t, el, rise.key))
        return out

    def _culmination(self, aos: float, los: float) -> tuple[float, float]:
        s = np.arange(np.ceil(aos), np.floor(los) + 1)
        if len(s) == 0:
            s = np.array([(aos + los) / 2])
        el = self.geometry(s).el_deg
        i = int(np.argmax(el))
        lo, hi = max(aos, float(s[i]) - 1.0), min(los, float(s[i]) + 1.0)
        if hi <= lo:
            return float(s[i]), float(el[i])
        # Search in time relative to lo: bounded Brent's tolerance is relative to |x| (27 s at Unix 1.8e9 s).
        r = minimize_scalar(
            lambda x: -float(self.geometry(lo + x).el_deg[0]),
            bounds=(0.0, hi - lo),
            method="bounded",
            options={"xatol": 1e-4},
        )
        return lo + float(r.x), -float(r.fun)

    def pass_keys(self, t_unix: np.ndarray, passes: list[Pass]) -> np.ndarray:
        """For each time, the aos_key of the pass containing it, or -1 (below the horizon)."""
        t = np.asarray(t_unix, dtype=np.float64)
        key = np.full(t.shape, -1, dtype=np.int64)
        for p in passes:
            key[(t >= p.aos) & (t < p.los)] = p.aos_key if p.aos_key is not None else -1
        return key
