"""Front end: noise FIR design, the band fallback, profile checks, ADC quantisation."""

import json
from pathlib import Path

import numpy as np
import pytest

from assist.config.loader import load_band
from assist.engine.frontend import FIR_TAPS, Frontend, NoiseSpec, design_fir, fir_response, noise_spec
from assist.rng import KeyedRng

ROOT = Path(__file__).resolve().parent.parent
PROFILE = ROOT / "profiles" / "vhf137"
FS = 2_400_000


def test_fir_follows_the_profile_shape() -> None:
    """513 taps by frequency sampling: 0.03 dB RMS from the profile's noise shape across 135.85-138.15 MHz."""
    shape = np.load(PROFILE / "noise_psd.npy").astype(np.float64)
    h = design_fir(shape)
    assert len(h) == FIR_TAPS
    np.testing.assert_allclose(h[::-1], np.conj(h), atol=1e-15)  # Hermitian: real, zero-phase response
    resp = np.fft.fftshift(fir_response(h, len(shape)))
    f = 137e6 + (np.arange(len(shape)) - len(shape) // 2) * FS / len(shape)
    band = (f > 135.85e6) & (f < 138.15e6)
    err = 20 * np.log10(resp[band]) - shape[band]
    assert np.sqrt(np.mean(err**2)) < 0.05


def test_flat_fir_is_a_delta() -> None:
    h = design_fir(None)
    assert h[FIR_TAPS // 2] == 1 and np.count_nonzero(h) == 1
    np.testing.assert_allclose(fir_response(h, 4800), 1.0)


def test_band_fallback_sigma_matches_profile() -> None:
    prof = json.loads((PROFILE / "profile.json").read_text())
    band = load_band("vhf137")
    assert band["noise"]["sigma_counts"] == pytest.approx(prof["adc"]["noise_sigma_mad_counts"], abs=6e-4)
    spec = noise_spec(PROFILE, band, 137e6, 2.4e6, None)
    assert spec.shape_db is not None and spec.sigma_counts == prof["adc"]["noise_sigma_mad_counts"]
    assert noise_spec(PROFILE, band, 137e6, 2.4e6, 20.0).sigma_counts == 20.0
    with pytest.raises(ValueError, match="profile"):
        noise_spec(PROFILE, band, 136e6, 2.4e6, None)


def test_noise_density_is_k_t_sys() -> None:
    """Flat noise: complex power per sample = k T_sys fs (in mW), and the ADC gain maps it to sigma counts."""
    t_sys = 727.0
    fe = Frontend(NoiseSpec(None, 26.7, "test"), t_sys, FS, FS // 100, FS // 50, KeyedRng(1).scope())
    y = np.concatenate([fe.noise(h) for h in range(1000, 1040)])
    p_mw = 1.380649e-23 * t_sys * FS * 1000
    assert np.mean(np.abs(y) ** 2) == pytest.approx(p_mw, rel=0.01)
    assert np.std((y * fe.gain).real) == pytest.approx(26.7, rel=0.01)


def test_noise_is_keyed_by_absolute_hop() -> None:
    a = Frontend(NoiseSpec(None, 26.7, "t"), 700.0, FS, FS // 100, FS // 50, KeyedRng(5).scope())
    b = Frontend(NoiseSpec(None, 26.7, "t"), 700.0, FS, FS // 100, FS // 50, KeyedRng(5).scope())
    seq = [a.noise(h) for h in range(7, 10)]
    np.testing.assert_array_equal(b.noise(9), seq[2])  # hop 9 on its own = hop 9 after 7 and 8
    c = Frontend(NoiseSpec(None, 26.7, "t"), 700.0, FS, FS // 100, FS // 50, KeyedRng(6).scope())
    assert not np.array_equal(c.noise(9), seq[2])


def test_adc_rounds_clips_and_interleaves() -> None:
    fe = Frontend(NoiseSpec(None, 26.7, "t"), 700.0, FS, FS // 100, FS // 50, KeyedRng(1).scope())
    g = fe.gain
    y = np.array([0, (1 + 2j) / g, (-200 + 300j) / g, (0.4 - 0.6j) / g], dtype=np.complex64)
    u = fe.adc(y)
    assert u.dtype == np.uint8
    assert u.tolist() == [128, 128, 128, 130, 0, 255, 128, 127]  # rint(127.5 + x): 127.5 -> 128 (half to even)
