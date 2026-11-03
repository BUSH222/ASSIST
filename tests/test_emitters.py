"""Link budget (plan §6.2), carrier model (§6.3), schedules and AM (§6.4): T-SCH."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from assist.config.loader import load_band
from assist.config.resolve import calibrations
from assist.emitters import link
from assist.emitters.carrier import Carrier, to_mhz
from assist.emitters.schedules import SLOT_S, AmModel, DurationDist, ExpGaps, OnOff, PoissonEvents
from assist.rng import KeyedRng

ROOT = Path(__file__).resolve().parent.parent
T0 = 1793551140.0  # 2026-11-01T16:39:00Z


# --- link budget ---


def test_fspl_and_patterns() -> None:
    assert float(link.fspl_db(1000e3, 137.5e6)) == pytest.approx(135.21, abs=0.01)
    qfh = link.Pattern()
    assert float(qfh.gain_db(12.0)) == pytest.approx(3.49, abs=0.01)
    assert float(qfh.gain_db(120.0)) == pytest.approx(float(qfh.gain_db(90.0)))  # clamped
    poly = link.Pattern("poly", coeffs_db=(1.0, 0.1))
    assert float(poly.gain_db(10.0)) == pytest.approx(2.0)
    assert float(poly.gain_db(100.0)) == pytest.approx(10.0)  # clamped to the 90 deg value
    card = link.Pattern("cardioid", n=2.0, peak_dbi=3.0)
    assert float(card.gain_db(0.0)) == pytest.approx(3.0)
    assert float(card.gain_db(90.0)) == pytest.approx(3.0 + 20 * math.log10(0.5))
    assert float(link.Pattern("isotropic", gain_dbi=-1.0).gain_db(33.0)) == -1.0


def test_cn0_zenith_round_trip() -> None:
    t_sys = link.t_sys_k(500, 1.465)
    qfh = link.Pattern()
    p = link.p_tx_for_cn0_zenith(45.0, 820e3, 137.5e6, qfh, qfh, t_sys)
    rx = link.p_rx_dbm(p, qfh.gain_db(0.0), np.array([820e3]), 137.5e6, qfh.gain_db(0.0))
    assert float(rx[0]) - 30 - link.kt_dbw_hz(t_sys) == pytest.approx(45.0)


def test_meteor_calibration_reproduces_m1_p_tx() -> None:
    """Link constants from meteor_cal.json give M1's P_tx at T_ant 300/500/1000 K."""
    mc = json.loads((ROOT / "profiles/vhf137/meteor_cal.json").read_text())
    cal, _ = calibrations(str(ROOT / "profiles/vhf137"), load_band("vhf137"))
    for t_ant, row in mc["p_tx_dbw_for_t_ant"].items():
        t_sys = link.t_sys_k(float(t_ant), mc["saw_loss_db_137_9"])
        # meteor_cal.json holds values rounded to 6 decimals
        assert link.p_tx_from_link_const(cal["meteor"], t_sys) == pytest.approx(row["both"], abs=1e-5)
        assert link.p_tx_from_link_const(cal["meteor_m2_3"], t_sys) == pytest.approx(row["METEOR-M2 3"], abs=1e-5)


def test_band_calibrations_match_profile() -> None:
    """The band's fallback calibrations are copies of the committed profile (to 0.001 dB)."""
    from_profile, _ = calibrations(str(ROOT / "profiles/vhf137"), load_band("vhf137"))
    from_band, _ = calibrations(None, load_band("vhf137"))
    assert set(from_band) == set(from_profile)
    for k, v in from_profile.items():
        assert from_band[k] == pytest.approx(v, abs=6e-4), k


# --- carrier ---


def test_carrier_per_pass_terms_are_keyed_by_aos() -> None:
    rng = KeyedRng(5)
    c = Carrier("z1", "carrier", to_mhz(137.5e6), 50.0, 0.2, rng)
    t = T0 + np.array([10.0, 20.0, 5000.0])
    keys = np.array([int(T0), int(T0), int(T0) + 4000])
    base, small = c.f_tx(t, keys)
    off0, drift0 = c.pass_terms(int(T0))
    assert base[0] == base[1] == to_mhz(137.5e6) + off0
    assert small[1] - small[0] == pytest.approx(drift0 * 10.0)
    assert abs(drift0) == 0.2
    # A fresh carrier (another window) draws the same terms for the same pass
    c2 = Carrier("z1", "carrier", to_mhz(137.5e6), 50.0, 0.2, KeyedRng(5))
    assert c2.pass_terms(int(T0) + 4000) == c.pass_terms(int(T0) + 4000)
    # Statistics over many passes: offset ~ N(0, 50 Hz), drift sign 50/50
    terms = [c.pass_terms(int(T0) + 6000 * k) for k in range(2000)]
    offs = np.array([o for o, _ in terms]) / 1000.0
    assert np.std(offs) == pytest.approx(50.0, rel=0.08)
    assert abs(np.mean([d > 0 for _, d in terms]) - 0.5) < 0.05
    # No oscillator: rest frequency only, below the horizon too
    plain = Carrier("s", "e", to_mhz(137.9e6))
    b, s = plain.f_tx(t, np.array([-1, 5, 7]))
    assert (b == to_mhz(137.9e6)).all() and (s == 0).all()


# --- schedules (T-SCH) ---


def test_on_off_is_exactly_40_on_180_off() -> None:
    sch = OnOff(40.0, 180.0, phase_s=123.456, ramp_ms=5.0)
    iv = sch.on_intervals(T0, T0 + 86400)
    lengths = np.array([b - a for a, b in iv])
    gaps = np.array([iv[i + 1][0] - iv[i][1] for i in range(len(iv) - 1)])
    np.testing.assert_allclose(lengths, 40.0, atol=1e-6)
    np.testing.assert_allclose(gaps, 180.0, atol=1e-6)
    assert all(
        (a - 123.456) % 220 == pytest.approx(0, abs=1e-6) or (a - 123.456) % 220 == pytest.approx(220, abs=1e-6)
        for a, _ in iv
    )
    a, b = iv[3]
    env = sch.envelope(np.array([a, b, a - 0.0026, a + 0.0026, (a + b) / 2, b + 0.0026, a - 1.0]))
    np.testing.assert_allclose(env[:2], 0.5, atol=1e-4)  # half amplitude at the edges (float64 time: ~0.2 us)
    np.testing.assert_allclose(env[2:], [0, 1, 1, 0, 0], atol=1e-6)
    frac = sch.active_fraction(np.arange(T0, T0 + 220 * 50, 1.0), 1.0)
    assert frac.sum() == pytest.approx(40.0 * 50, abs=1e-6)
    assert set(np.round(frac, 6)) <= {0.0, 1.0, 0.544, 0.456}  # partial seconds at the edges (phase .456)


def _durations(d: DurationDist, n: int = 20000) -> np.ndarray:
    return d.draw(np.random.default_rng(1), n)


def test_duration_distributions() -> None:
    ln = _durations(DurationDist("lognormal", median_s=4.0, sigma=0.74, max_s=1e6))
    assert np.median(ln) == pytest.approx(4.0, rel=0.03)
    assert np.std(np.log(ln)) == pytest.approx(0.74, rel=0.03)
    assert _durations(DurationDist("exponential", mean_s=3.0, max_s=5.0)).max() <= 5.0
    assert (_durations(DurationDist("fixed", s=2.5)) == 2.5).all()


def test_exp_gaps_statistics_and_random_access() -> None:
    rng = KeyedRng(11)
    sch = ExpGaps(rng, "bee", "lora", DurationDist("fixed", s=1.0), mean_gap_s=180.0, min_gap_s=20.0)
    ev = sch.events(T0, T0 + 30 * 86400)
    starts = np.array([e.start for e in ev])
    ends = np.array([e.end for e in ev])
    gaps = starts[1:] - ends[:-1]
    assert gaps.min() >= 20.0 - 1e-9  # also across hour slots
    assert np.mean(gaps) == pytest.approx(180.0, rel=0.05)
    # Any window gives the same events: a fresh schedule queried over a sub-window
    sub = ExpGaps(KeyedRng(11), "bee", "lora", DurationDist("fixed", s=1.0)).events(T0 + 5000.5, T0 + 9000)
    assert sub == [e for e in ev if e.end > T0 + 5000.5 and e.start < T0 + 9000]
    e0 = ev[0]
    env = sch.envelope(np.array([e0.start - 0.5, e0.start, (e0.start + e0.end) / 2, e0.end]))
    np.testing.assert_array_equal(env, [0, 1, 1, 0])
    with pytest.raises(ValueError, match="min_gap_s"):
        ExpGaps(rng, "a", "b", DurationDist("fixed", s=30.0), 180.0, 20.0)


def test_poisson_events_rate_curve_and_random_access() -> None:
    curve = [10.0] * 12 + [40.0] * 12
    sch = PoissonEvents(
        KeyedRng(3), "vdl", "ch1", DurationDist("lognormal", median_s=4.0, sigma=0.74, max_s=600), curve
    )
    day0 = np.floor(T0 / 86400) * 86400
    ev = sch.events(day0, day0 + 40 * 86400)
    hours = np.array([int((e.start % 86400) // 3600) for e in ev])
    per_h = np.bincount(hours, minlength=24) / 40
    assert per_h[:12].mean() == pytest.approx(10.0, rel=0.1)
    assert per_h[12:].mean() == pytest.approx(40.0, rel=0.1)
    sub = PoissonEvents(KeyedRng(3), "vdl", "ch1", sch.duration, curve).events(
        day0 + 3 * SLOT_S + 7, day0 + 9 * SLOT_S
    )
    assert sub == [e for e in ev if e.end > day0 + 3 * SLOT_S + 7 and e.start < day0 + 9 * SLOT_S]
    # Different emitters (keys) give different events
    other = PoissonEvents(KeyedRng(3), "vdl", "ch2", sch.duration, curve).events(day0, day0 + 86400)
    assert other != [e for e in ev if e.start < day0 + 86400]


def test_am_tumbling_gain() -> None:
    am = AmModel(7.5, 6.0, 0.3)
    t = T0 + np.linspace(0, 30, 3001)
    g = am.gain_db(t)
    assert g.max() == pytest.approx(0.0, abs=1e-3) and g.min() == pytest.approx(-6.0, abs=1e-3)
    np.testing.assert_allclose(am.gain_db(t + 7.5), g, atol=1e-9)
    ref = -(6.0 / 2) * (1 - np.cos(2 * np.pi * (T0 % 7.5) / 7.5 + 0.3))
    assert float(am.gain_db(np.array([T0]))[0]) == pytest.approx(ref)
