"""Orbit modes and spline geometry: Doppler (truth vs Skyfield), synthetic passes, perturbation."""

import datetime as dt
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from assist.emitters.link import C_M_S
from assist.orbits import elements as E
from assist.orbits.geometry import (
    BLOCK_S,
    MARGIN_S,
    Geometry,
    SplineGeometry,
    sgp4_topocentric,
    station_frame,
    topocentric,
)
from assist.orbits.perturb import perturb_omm
from assist.orbits.synthetic import RE_KM, Kepler, PassRequest, kepler_satellite, kepler_satrec, solve_pass
from assist.station import Station

FIXTURE = Path(__file__).parent / "fixtures" / "omm_2026-11-02.csv"
STATION = Station(55.5, 37.1, 172.0)
T_REC = dt.datetime(2026, 11, 1, 16, 39, tzinfo=dt.UTC).timestamp()
METEOR_M2_3 = 57166


@pytest.fixture(scope="module")
def meteor() -> SplineGeometry:
    s = E.ElementStore.load([FIXTURE]).nearest(METEOR_M2_3, T_REC)
    return SplineGeometry(E.satellite_from_omm(s.omm), STATION, 0.0, METEOR_M2_3, "METEOR-M2 3")


def test_truth_doppler_matches_skyfield(meteor: SplineGeometry) -> None:
    """Doppler in truth: spline range rate vs Skyfield's own value, as Doppler at 137.9 MHz: <= 0.01 Hz.

    Geometry requirements against full Skyfield: range within 1 m, angles within 1e-2 deg.
    """
    ps = meteor.passes(T_REC, T_REC + 86400)
    assert len(ps) >= 6
    rng = np.random.default_rng(0)
    t = np.concatenate([rng.uniform(p.aos, p.los, 200) for p in ps])
    t = np.concatenate([t, np.array([np.ceil(T_REC / BLOCK_S) * BLOCK_S + d for d in (-0.5, 0.0, 0.5)])])
    g = meteor.geometry(t)
    ref = topocentric(meteor.sat, STATION, t)  # full IAU 2000A model, no splines
    dop = -137.9e6 * (g.range_rate_m_s - ref.range_rate_m_s) / C_M_S
    assert np.max(np.abs(dop)) <= 0.01
    _assert_close_to_skyfield(g, ref)


def _assert_close_to_skyfield(g: Geometry, ref: Geometry) -> None:
    assert np.max(np.abs(g.range_m - ref.range_m)) < 1.0  # m
    assert np.max(np.abs(g.el_deg - ref.el_deg)) < 1e-2
    az_err = np.abs((g.az_deg - ref.az_deg + 180) % 360 - 180)
    assert np.max(az_err[ref.el_deg < 85]) < 1e-2
    assert np.max(np.abs(g.off_nadir_deg - ref.off_nadir_deg)) < 1e-2


@pytest.mark.parametrize(
    "kep",
    [
        Kepler(T_REC - 86400.0, 26_560.0, 0.01, 55.0, 40.0, 30.0, 200.0),  # GPS-like (12 h, deep-space SGP4)
        Kepler(T_REC - 86400.0, 26_600.0, 0.72, 63.4, 300.0, 270.0, 120.0),  # Molniya-like
        Kepler(T_REC - 3600.0, RE_KM + 550.0, 0.0, 53.0, 120.0, 0.0, 10.0, 1e-4),  # LEO with drag
    ],
    ids=["gps", "molniya", "leo"],
)
def test_direct_sgp4_matches_skyfield(kep: Kepler) -> None:
    """Direct SGP4 (the simulator's sampler) against full Skyfield over a day, above the horizon."""
    sat = kepler_satellite(kep)
    t = T_REC + np.arange(0.0, 86400.0, 7.3)
    ref = topocentric(sat, STATION, t)
    up = ref.el_deg > 0
    assert up.sum() > 100
    t0 = int(T_REC)
    g = sgp4_topocentric(sat.model, station_frame(STATION), t0, t[up] - t0)
    sel = Geometry(
        *(np.asarray(getattr(ref, f))[up] for f in ("el_deg", "az_deg", "range_m", "range_rate_m_s", "off_nadir_deg"))
    )
    _assert_close_to_skyfield(g, sel)
    assert np.max(np.abs(g.range_rate_m_s - sel.range_rate_m_s)) < 1e-3  # m/s, both from SGP4's velocity
    spline = SplineGeometry(sat, STATION, 0.0).geometry(t[up])
    _assert_close_to_skyfield(spline, sel)


def test_blocks_agree_at_their_boundary(meteor: SplineGeometry) -> None:
    """Neighbouring hour blocks share the samples at their overlap (UT1 - UTC per sample, not per block)."""
    edge = int(np.ceil(T_REC / BLOCK_S)) * BLOCK_S
    s = np.arange(-MARGIN_S, MARGIN_S + 1, dtype=np.float64)
    a = sgp4_topocentric(meteor.sat.model, station_frame(STATION), edge - BLOCK_S, s + BLOCK_S)
    b = sgp4_topocentric(meteor.sat.model, station_frame(STATION), edge, s)
    np.testing.assert_allclose(a.range_m, b.range_m, rtol=0, atol=1e-6)


def test_geometry_is_window_independent(meteor: SplineGeometry) -> None:
    t = T_REC + np.arange(0.0, 7200.0, 0.37)
    whole = meteor.geometry(t)
    fresh = SplineGeometry(meteor.sat, STATION, 0.0)
    part = fresh.geometry(t[5000:])
    np.testing.assert_array_equal(whole.range_m[5000:], part.range_m)
    np.testing.assert_array_equal(whole.range_rate_m_s[5000:], part.range_rate_m_s)


def test_passes_and_keys_do_not_depend_on_the_window(meteor: SplineGeometry) -> None:
    full = meteor.passes(T_REC, T_REC + 86400)
    for p in full:
        assert p.aos < p.tca < p.los and 0 < p.max_el_deg <= 90
        assert p.aos_key is not None and p.aos_key == math.floor(p.aos + 0.5)  # nearest whole second
        mid = SplineGeometry(meteor.sat, STATION, 0.0, METEOR_M2_3, "METEOR-M2 3").passes(p.tca, p.tca + 1)
        assert mid == [p]  # window starting mid-pass finds the same pass and key
        g = meteor.geometry(np.array([p.aos, p.los]))
        np.testing.assert_allclose(g.el_deg, 0, atol=1e-6)


@pytest.mark.parametrize("direction", ["asc", "desc"])
@pytest.mark.parametrize("side", ["east", "west"])
@pytest.mark.parametrize("max_el", [3.0, 20.0, 45.0, 85.0])
def test_synthetic_pass_hits_aos_and_max_el(direction: str, side: str, max_el: float) -> None:
    """Done-when: synthetic pass hits the requested AOS +- 1 s and max elevation +- 0.5 deg."""
    aos = T_REC + 7 * 3600 + 123.0
    raan, m = solve_pass(PassRequest(aos, max_el, direction, side), STATION)  # ty: ignore[invalid-argument-type]
    sat = kepler_satellite(Kepler(aos, RE_KM + 820.0, 0.0, 98.7, raan, 0.0, m))
    g = SplineGeometry(sat, STATION, 0.0)
    (p,) = [p for p in g.passes(aos - 1800, aos + 1800) if abs(p.aos - aos) < 300]
    assert p.aos == pytest.approx(aos, abs=1.0)
    assert p.aos_key == aos  # a whole-second AOS is its own key, whichever side of it the solved AOS lands
    assert p.max_el_deg == pytest.approx(max_el, abs=0.5)
    tca = g.geometry(np.array([p.tca - 1.0, p.tca + 1.0]))
    lat_rate = sat.at(E.timescale().from_datetime(dt.datetime.fromtimestamp(p.tca, dt.UTC)))
    assert (np.asarray(lat_rate.velocity.km_per_s)[2] > 0) == (direction == "asc")
    az = float(g.geometry(np.array([p.tca])).az_deg[0])
    assert (0 < az < 180) == (side == "east")
    assert tca.el_deg.max() <= p.max_el_deg + 1e-6


def test_synthetic_pass_rejects_impossible_geometry() -> None:
    with pytest.raises(ValueError):
        solve_pass(PassRequest(T_REC, 45.0, "asc", "east", 820.0, 20.0), STATION)  # 20 deg orbit, 55.5 N
    with pytest.raises(ValueError):
        solve_pass(PassRequest(T_REC, 0.0, "asc", "east"), STATION)


def test_kepler_mean_motion_and_epoch() -> None:
    k = Kepler(T_REC, RE_KM + 820.0, 0.001, 98.7, 10.0, 20.0, 30.0)
    s: Any = kepler_satrec(k, 91234)  # sgp4 Satrec (C or Python implementation)
    assert s.satnum == 91234
    assert (s.jdsatepoch + s.jdsatepochF - 2440587.5) * 86400 == pytest.approx(T_REC, abs=1e-3)
    period_min = 2 * np.pi / s.no_kozai
    assert period_min == pytest.approx(101.0, abs=0.5)  # 820 km LEO


def test_perturbation_along_track_is_a_time_shift() -> None:
    """along_track_s = 2: the true satellite is where the published one was 2 s earlier (inertial frame)."""
    s = E.ElementStore.load([FIXTURE]).nearest(METEOR_M2_3, T_REC)
    pub = E.satellite_from_omm(s.omm)
    true = E.satellite_from_omm(perturb_omm(s.omm, along_track_s=2.0))
    ts = E.timescale()
    t = ts.utc(2026, 11, 1, 18, 0, 0.0)
    t_early = ts.utc(2026, 11, 1, 18, 0, -2.0)
    d = np.asarray(true.at(t).position.km) - np.asarray(pub.at(t_early).position.km)
    assert np.linalg.norm(d) < 0.05  # km
    p2 = perturb_omm(s.omm, raan_deg=0.5, inc_deg=-0.1, dn_rev_day=0.01)
    assert float(p2["RA_OF_ASC_NODE"]) == pytest.approx((float(s.omm["RA_OF_ASC_NODE"]) + 0.5) % 360)
    assert float(p2["INCLINATION"]) == pytest.approx(float(s.omm["INCLINATION"]) - 0.1)
    assert float(p2["MEAN_MOTION"]) == pytest.approx(float(s.omm["MEAN_MOTION"]) + 0.01)
