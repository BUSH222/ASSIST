"""Profile models and statistics: QFH/link budget (plan §6.2), ADC, noise, birdies, occupancy, ORBCOMM, labels."""

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.signal import fftconvolve

from assist import waterfall as wf
from assist.profile import adc, birdies, labels, meteor_cal, noise, occupancy, orbcomm
from assist.profile.annotations import excluded_ranges, keep_mask, load_annotations

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("x", "g"), [(0, 3.37), (12, 3.49), (30, 3.18), (45, 2.39), (60, 1.05), (62.4, 0.77), (90, -3.63)]
)
def test_qfh_gain_matches_plan_table(x: float, g: float) -> None:
    assert meteor_cal.qfh_gain_db(np.array([x]))[0] == pytest.approx(g, abs=0.01)


def test_qfh_clamped_outside_domain() -> None:
    g90 = meteor_cal.qfh_gain_db(np.array([90.0]))[0]
    np.testing.assert_allclose(meteor_cal.qfh_gain_db(np.array([-5.0, 120.0])), g90)


def test_link_budget_geometry_matches_plan() -> None:
    """820 km orbit: off-nadir 62.4 deg at the horizon; zenith vs horizon ~ -21.8 dB in total (plan §6.2)."""
    h, re = 820e3, meteor_cal.R_EARTH
    r_horizon = np.sqrt((re + h) ** 2 - re**2)
    assert meteor_cal.off_nadir_deg(np.array([0.0]), np.array([r_horizon]), 0.0)[0] == pytest.approx(62.4, abs=0.2)
    zen = meteor_cal.model_snr_shape_db(np.array([90.0]), np.array([h]), 0.0)[0]
    hor = meteor_cal.model_snr_shape_db(np.array([0.0]), np.array([r_horizon]), 0.0)[0]
    assert hor - zen == pytest.approx(-21.8, abs=0.3)


def test_band_fraction_and_tsys() -> None:
    assert 0.98 < meteor_cal.band_power_fraction() <= 1.0
    # T_ant only, lossless SAW, ideal LNA/tuner: T_sys = T_ant
    assert meteor_cal.t_sys_k(500, 0.0, lna_nf_db=0.0, tuner_nf_db=0.0) == pytest.approx(500)
    assert meteor_cal.t_sys_k(500, 1.47) == pytest.approx(727, abs=2)


def test_adc_histogram_moments() -> None:
    rng = np.random.default_rng(0)
    u = np.clip(np.round(127.5 + 25 * rng.standard_normal(200_000)), 1, 254).astype(np.uint8)
    h = np.bincount(u, minlength=256)[None, :]
    m = adc.hist_moments(h)
    assert m["sigma_mad"][0] == pytest.approx(25, rel=0.03)
    assert m["rms"][0] == pytest.approx(25, rel=0.02)
    assert m["clip"][0] == pytest.approx(np.mean((u <= 1) | (u >= 254)))
    assert adc.image_rejection_db(1.01, 0.01) == pytest.approx(-43.0, abs=0.1)


def test_dc_amplitude_estimate() -> None:
    rng = np.random.default_rng(1)
    sigma, dc = 26.0, 0.8 + 0.6j  # counts
    n = wf.ROW_SAMPLES
    x = (sigma * (rng.standard_normal(n) + 1j * rng.standard_normal(n)) + dc) / 127.5
    p = wf.power_db(wf.mean_power(x.astype(np.complex64)))
    exc = noise.dc_spike(p)["centre_bin_excess_db"]
    assert noise.dc_amplitude_counts(exc, sigma) == pytest.approx(abs(dc), rel=0.25)


def test_combined_shape_splices_clipped_bins() -> None:
    f = wf.bin_freqs(137.0e6) / 1e6
    true = -8 * np.exp(-(((f - 135.95) / 0.05) ** 2))  # a notch
    day = np.maximum(true + 1.0, noise.FLOOR_DB)  # clipped at the waterfall floor
    shape, clipped = noise.combined_shape([day, day], true, f)
    assert clipped.any()
    np.testing.assert_allclose(shape[clipped], noise.normalise(true, f)[clipped], atol=0.05)


def test_birdies_need_persistence() -> None:
    rng = np.random.default_rng(2)
    f = wf.bin_freqs(137.0e6)
    m = 0.05 * rng.standard_normal((200, wf.FFT_SIZE))
    m[:, 3000] += 2.0  # persistent
    m[:100, 6000] += 2.0  # half of the time
    b = birdies.find_birdies(m, f)
    assert [x["freq_hz"] for x in b] == [f[3000]]
    assert b[0]["excess_db_median"] == pytest.approx(2.0, abs=0.1)


def test_regimes_two_levels() -> None:
    lv = np.r_[np.full(300, 1.0), np.full(200, 2.5), np.full(300, 1.0)]
    lv += 0.05 * np.random.default_rng(3).standard_normal(800)
    r = noise.regimes(lv)
    assert r["loud_fraction"] == pytest.approx(0.25, abs=0.02)
    assert r["loud_level_db"] - r["quiet_level_db"] == pytest.approx(1.5, abs=0.1)


def test_occupancy_bursts_and_fits() -> None:
    rng = np.random.default_rng(4)
    frame = 1024 / 2.4e6
    p = np.ones(200_000) * (1 + 0.1 * rng.random(200_000))
    durs = rng.lognormal(np.log(0.022), 0.2, 300)
    t = 1000
    for d in durs:
        k = int(d / frame)
        p[t : t + k] = 100.0
        t += k + 400
    b = occupancy.bursts(p, frame)
    assert b["events"] == 300
    assert b["duration_s"]["median"] == pytest.approx(np.median(durs), rel=0.1)
    ln = occupancy.lognormal_fit(durs)
    assert ln["sigma"] == pytest.approx(0.2, abs=0.03)
    chans = occupancy.default_channels()
    assert sum(c.kind == "vdl" for c in chans) == 12
    assert [c.kind for c in chans].count("band138") == 1


def _rrc(alpha: float, sps: int, span: int) -> np.ndarray:
    t = np.arange(-span * sps, span * sps + 1) / sps
    h = np.empty_like(t)
    for i, x in enumerate(t):
        if x == 0:
            h[i] = 1 - alpha + 4 * alpha / np.pi
        elif abs(abs(x) - 1 / (4 * alpha)) < 1e-9:
            h[i] = (
                alpha
                / np.sqrt(2)
                * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha)))
            )
        else:
            h[i] = (np.sin(np.pi * x * (1 - alpha)) + 4 * alpha * x * np.cos(np.pi * x * (1 + alpha))) / (
                np.pi * x * (1 - (4 * alpha * x) ** 2)
            )
    return h / np.sqrt(np.sum(h**2))


def test_orbcomm_carrier_symbol_rate_and_alpha() -> None:
    """Synthetic +-90 deg DPSK (plan §5 sdpsk model), 5000 sym/s at 50 kS/s, carrier +350 Hz, alpha 0.4."""
    rng = np.random.default_rng(5)
    rs, sps, n_sym = 5000, 10, 5000 * 8
    p = (np.arange(n_sym) % 2) + 2 * rng.integers(0, 2, n_sym)
    sym = np.exp(1j * np.pi / 2 * p)
    up = np.zeros(n_sym * sps, complex)
    up[::sps] = sym
    x = fftconvolve(up, _rrc(0.4, sps, 8), mode="same")
    t = np.arange(len(x)) / orbcomm.RATE
    x = x * np.exp(2j * np.pi * 350 * t) + 0.05 * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    x = x.astype(np.complex64)
    tr = orbcomm.carrier_track(x, np.full(len(x) // orbcomm.RATE, 300.0), rs=rs)
    np.testing.assert_allclose(tr["offset_hz"], 350.0, atol=1.0)
    xc = orbcomm.derotate(x, tr["t"], tr["offset_hz"])
    assert orbcomm.symbol_rate(xc, guess_hz=rs) == pytest.approx(rs, abs=1.0)
    assert orbcomm.fit_pulse(xc, rs)["alpha"] == pytest.approx(0.4, abs=0.05)


def test_annotations(tmp_path: Path) -> None:
    a = tmp_path / "a.yaml"
    a.write_text("- {file: samples/x.wav, from_s: 10, to_s: 20, label: orbcomm, exclude_from: [noise, occupancy]}\n")
    b = tmp_path / "b.yaml"
    b.write_text("- {file: x.wav, from_s: 0, to_s: 5, label: local_interference, exclude_from: [noise]}\n")
    ann = load_annotations([a, b])
    assert excluded_ranges(ann, "x.wav", "noise") == [(10.0, 20.0), (0.0, 5.0)]
    assert excluded_ranges(ann, "x.wav", "occupancy") == [(10.0, 20.0)]
    assert excluded_ranges(ann, "y.wav", "noise") == []
    keep = keep_mask(np.array([2.0, 7.0, 15.0, 25.0]), [(10.0, 20.0), (0.0, 5.0)])
    assert keep.tolist() == [False, True, False, True]
    bad = tmp_path / "c.yaml"
    bad.write_text("- {file: x.wav, from_s: 0, to_s: 5, label: z, exclude_from: [everything]}\n")
    with pytest.raises(ValueError, match="unknown exclude_from"):
        load_annotations([bad])


def test_label_sampling_and_satband_events() -> None:
    ev = [{"id": f"e{i}", "auto_label": "vdl" if i % 10 else "am"} for i in range(1000)]
    s1 = labels.stratified_sample(ev, 20)
    assert s1 == labels.stratified_sample(ev, 20)
    assert {e["auto_label"] for e in s1} == {"vdl", "am"}
    f = wf.bin_freqs(137.0e6)
    mx = np.full((120, wf.FFT_SIZE), 60, np.uint8)  # ~2 dB
    k = int(np.argmin(np.abs(f - 137.3e6)))
    mx[30:33, k : k + 3] = 200  # ~ +18.6 dB
    base = np.full((2, wf.FFT_SIZE), 2.0)
    found = labels.satband_events(mx, base, f, np.zeros(mx.shape, bool))
    assert len(found) == 1
    m0, m1, b0, b1, _pk = found[0]
    assert (m0, m1, b0, b1) == (30, 33, k, k + 3)
