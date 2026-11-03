"""The IQ measurements used by the render tests (assist.measure), on synthetic tones with known answers."""

import numpy as np
import pytest

from assist import measure

FS = 2_400_000


def _chirp(n: int, f0: float, rate_hz_s: float, cn0_dbhz: float, seed: int = 0) -> np.ndarray:
    """A tone f0 + rate * t (t from the window centre) in complex white noise of density 1 per Hz."""
    t = (np.arange(n) - n / 2) / FS
    a = np.sqrt(10 ** (cn0_dbhz / 10))  # carrier power C = cn0 * N0 with N0 = 1
    g = np.random.default_rng(seed)
    noise = (g.standard_normal(n) + 1j * g.standard_normal(n)) * np.sqrt(FS / 2)
    return a * np.exp(2j * np.pi * (f0 * t + rate_hz_s * t**2 / 2)) + noise


@pytest.mark.parametrize(("f0", "rate"), [(350_123.4, -25.0), (-1_097_046.7, 0.4), (0.3, 12.0)])
def test_tone_freq_at_the_window_centre(f0: float, rate: float) -> None:
    y = _chirp(FS // 2, f0, rate, 40.0)
    got = measure.tone_freq(y, FS, f0 + 13.0, half_width=60.0)  # guess 13 Hz off
    assert got == pytest.approx(f0, abs=0.05)


@pytest.mark.parametrize("cn0", [40.0, 50.0, 60.0])
def test_cn0(cn0: float) -> None:
    """One 1 s window: the estimate scatters by ~0.08 dB (1 sigma, mostly the N0 estimate), so 0.3 dB is ~3.5 sigma."""
    y = _chirp(FS, 123_456.0, -20.0, cn0, seed=int(cn0))
    got, _c, n0 = measure.cn0_dbhz(y, FS, 123_456.0 - 30, 123_456.0 + 30, ring=(300, 3000))
    assert got == pytest.approx(cn0, abs=0.3)
    assert n0 == pytest.approx(1.0, rel=0.05)


def test_cn0_is_unbiased() -> None:
    d = []
    for s in range(8):
        y = _chirp(FS // 2, -54_321.0, 5.0, 45.0, seed=100 + s)
        d.append(measure.cn0_dbhz(y, FS, -54_321.0 - 30, -54_321.0 + 30, ring=(300, 3000))[0] - 45.0)
    assert abs(np.mean(d)) <= 0.08


def test_inst_freq_and_u8() -> None:
    n = np.arange(10_000)
    y = np.exp(2j * np.pi * 350_000.25 * n / FS).astype(np.complex64)
    assert np.mean(measure.inst_freq(y, FS)) == pytest.approx(350_000.25, abs=0.01)
    u = np.array([0, 255, 128, 127], dtype=np.uint8)
    np.testing.assert_array_equal(measure.iq_from_u8(u), [-127.5 + 127.5j, 0.5 - 0.5j])
