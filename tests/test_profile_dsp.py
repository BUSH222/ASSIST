"""Profile signal-processing helpers: channelizer, raised cosine, runs/hysteresis, image tests, comb."""

import numpy as np
import pytest
from scipy.signal import upfirdn

from assist import waterfall as wf
from assist.profile import humps, mirror
from assist.profile.dsp import channelize, hysteresis, lowpass_taps, rc_psd, runs

FS = 2_400_000.0


def _signal(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = 0.5 * np.exp(2j * np.pi * 123_456.7 * t / FS) + 0.3 * np.exp(-2j * np.pi * 400_000 * t / FS)
    return (x + 0.05 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(np.complex64)


def test_channelize_matches_direct_and_is_chunk_invariant() -> None:
    x = _signal(200_000)

    def read(a: int, k: int) -> np.ndarray:
        return x[a : a + k]

    n0, n1, off, d = 30_000, 150_000, 123_456.7, 48
    taps = lowpass_taps(FS, d)
    got = channelize(read, n0, n1, FS, off, d, taps, chunk=17 * d * 100)
    got2 = channelize(read, n0, n1, FS, off, d, taps, chunk=10**9)
    np.testing.assert_allclose(got, got2, rtol=0, atol=1e-6)
    # Direct: mix by absolute index, full convolution, output m centred on input n0 + m*d
    idx = np.arange(len(x))
    y = np.convolve(x.astype(np.complex128) * np.exp(-2j * np.pi * off * idx / FS), taps)
    delay = (len(taps) - 1) // 2
    want = y[n0 + delay : n1 + delay : d]
    np.testing.assert_allclose(got, want[: len(got)], rtol=0, atol=1e-5)
    # The tone at the mixing frequency comes out as a constant of amplitude 0.5
    assert np.median(np.abs(got)) == pytest.approx(0.5, abs=0.01)
    assert np.std(np.angle(got)) < 0.2


def test_upfirdn_convention() -> None:
    """channelize relies on upfirdn(h, x, down=D)[j] == full_conv(h, x)[j*D]."""
    rng = np.random.default_rng(1)
    h, x = rng.standard_normal(33), rng.standard_normal(500)
    np.testing.assert_allclose(upfirdn(h, x, down=4), np.convolve(x, h)[::4][: len(upfirdn(h, x, down=4))])


def test_rc_psd() -> None:
    f = np.array([0.0, 1400.0, 2400.0, 3360.0, 3400.0, -2400.0])
    p = rc_psd(f, 4800.0, 0.4)
    assert p[0] == 1.0
    assert p[1] == pytest.approx(1.0)  # (1 - alpha) rs / 2 = 1440
    assert p[2] == pytest.approx(0.5)  # rs / 2
    assert p[3] == pytest.approx(0.0, abs=1e-12)  # (1 + alpha) rs / 2 = 3360
    assert p[4] == 0.0
    assert p[5] == p[2]


def test_runs_and_hysteresis() -> None:
    x = np.array([0, 5, 9, 7, 4, 2, 0, 4, 4, 0, 11, 0], dtype=float)
    on = hysteresis(x, on_th=8, off_th=3)
    assert on.tolist() == [False, True, True, True, True, False, False, False, False, False, True, False]
    assert runs(on).tolist() == [[1, 5], [10, 11]]


def _iq(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)


@pytest.mark.parametrize("coherent", [True, False])
def test_coherent_image(coherent: bool) -> None:
    n = 600_000
    t = np.arange(n)
    rng = np.random.default_rng(3)
    # A bursty narrowband source at -75 kHz and either its image (beta) or independent activity at +75 kHz
    env = np.repeat(rng.random(n // 6000) > 0.5, 6000).astype(float)
    # Each transmitter has its own random phase walk (modulation / oscillator), so independent ones are incoherent.
    src = env * np.exp(1j * (-2 * np.pi * 75e3 * t / FS + np.cumsum(0.05 * rng.standard_normal(n))))
    beta = 0.1 * np.exp(0.7j)
    other = np.roll(env, 30_000) * np.exp(1j * (2 * np.pi * 75e3 * t / FS + np.cumsum(0.05 * rng.standard_normal(n))))
    y = src + (beta * np.conj(src) if coherent else 0.1 * other) + 0.03 * _iq(n, 4)

    def read(a: int, k: int) -> np.ndarray:
        return y[a : a + k].astype(np.complex64)

    r = mirror.coherent_image(read, 0, n, FS, -75e3)
    if coherent:
        assert r["beta_db"] == pytest.approx(-20.0, abs=0.5)
        assert r["coherence"] > 0.8
    else:
        assert r["coherence"] < 0.01


def test_image_rejection_minute_scale() -> None:
    f = wf.bin_freqs(137.0e6)
    rng = np.random.default_rng(5)
    n = 300
    spectra = 0.05 * rng.standard_normal((n, wf.FFT_SIZE))
    src = np.flatnonzero(np.abs(f - 136.925e6) <= 10e3)
    level = rng.uniform(0, 30, n)  # linear excess over a unit noise floor
    lin = 1 + np.zeros((n, wf.FFT_SIZE))
    lin[:, src] += level[:, None]
    lin[:, mirror.mirror_bins(src)] += 0.01 * level[:, None]
    spectra += 10 * np.log10(lin)
    r = mirror.image_rejection(spectra, f, 136.925e6, 10e3)
    assert r["image_db"] == pytest.approx(-20.0, abs=1.0)


def test_comb_series_recovers_spacing_and_amplitude() -> None:
    fcol = wf.bin_freqs(137.0e6).reshape(-1, humps.COL_BINS).mean(axis=1) / 1e6
    steps = 20
    cube = np.zeros((steps, len(fcol)), np.float32)
    for s in range(steps):
        cube[s] = 0.3 * np.cos(2 * np.pi * (fcol * 1e6) / 80e3 + 0.3 * s)
    cs = humps.comb_series(cube, fcol)
    assert np.median(cs["spacing_hz"]) == pytest.approx(80e3, rel=0.03)
    assert np.median(cs["amp_db"]) == pytest.approx(0.3, rel=0.25)
    summ = humps.comb_summary(cs, 10.0)
    assert summ["fraction_on"] == 1.0
    # 0.3 rad per 10 s step at 80 kHz spacing: 0.3 / (2 pi) * 80e3 / 10 = 382 Hz/s
    assert summ["wander_hz_s_p50_p90"][0] == pytest.approx(382, rel=0.1)
