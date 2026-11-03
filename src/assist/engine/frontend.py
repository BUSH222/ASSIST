"""Receiver front end, M3 part (plan §4.3, §4.4): profile-shaped noise, the receiver response and the ADC.

- **Noise shape.** The profile's noise PSD (`noise_psd.npy`: 16 384 fftshifted bins, dB, 0 dB = passband median) is
  turned into a 513-tap FIR by frequency sampling: the inverse FFT of the amplitude 10^(dB/20) with zero phase,
  truncated to the central 513 taps. The taps are Hermitian, so the response is real (zero phase). Without a profile
  the shape is flat (a single tap).
- **Noise.** White complex Gaussian at fs, one keyed draw per absolute hop ("noise", hop) (plan §3.2), filtered by
  overlap-save. Hop h uses its own draw plus the last 512 samples of hop h-1's draw, so any hop can be computed on
  its own. The PSD's 0 dB level is k T_sys: noise and signals are both referred to the antenna terminals in sqrt(mW).
- **Receiver response for signals** (stage 1, default): the same FIR's magnitude on the synthesiser's bin grid. Signals
  and noise then share one shape, so the calibrated C/N0 holds at every frequency.
- **ADC** (stage 7): u = clip(rint(127.5 + G * I/Q), 0, 255), interleaved I, Q. G is set so the shaped noise has
  `noise_sigma_counts` per component (profile: 26.7, the robust estimate from the recording).
"""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.fft as sfft

from assist.emitters.link import K_BOLTZ

if TYPE_CHECKING:
    from pathlib import Path

    from assist.rng import KeyedRng

FIR_TAPS = 513
PROFILE_FFT = 16384


@dataclass(frozen=True)
class NoiseSpec:
    """Where the noise comes from: a shape (fftshifted dB over fs, or None = flat) and the ADC noise sigma."""

    shape_db: np.ndarray | None
    sigma_counts: float
    source: str


def noise_spec(
    profile_dir: Path | None,
    band: dict[str, Any],
    fc_hz: float,
    fs_hz: float,
    sigma_override: float | None,
    name: str | None = None,
) -> NoiseSpec:
    """The profile's noise shape and sigma, else a flat shape with the band's sigma. An explicit sigma wins.

    `name` is how the profile is reported (the scenario's relative path; default: profile_dir).
    """
    if profile_dir is not None:
        prof = json.loads((profile_dir / "profile.json").read_text())
        if float(prof["fc_hz"]) != fc_hz or float(prof["fs_hz"]) != fs_hz:
            raise ValueError(
                f"profile {profile_dir} is for fc {prof['fc_hz']} Hz, fs {prof['fs_hz']} Hz; "
                f"the scenario has fc {fc_hz} Hz, fs {fs_hz} Hz"
            )
        shape = np.load(profile_dir / prof["noise"]["psd_file"]).astype(np.float64)
        sigma = float(prof["adc"]["noise_sigma_mad_counts"])
        source = f"profile {name or profile_dir}"
    else:
        shape = None
        sigma = float(band["noise"]["sigma_counts"])
        source = f"band {band.get('name', '')} (flat)"
    if sigma_override is not None:
        sigma = sigma_override
        source += ", sigma from frontend.noise_sigma_counts"
    return NoiseSpec(shape, sigma, source)


def design_fir(shape_db: np.ndarray | None, taps: int = FIR_TAPS) -> np.ndarray:
    """Zero-phase FIR (complex128, `taps` long, centre tap at taps // 2) whose magnitude follows shape_db.

    Frequency sampling: inverse FFT of the amplitude on the shape's own grid, truncated to the central taps without a
    taper (the shape is smooth: 0.03 dB RMS from the profile across 135.85-138.15 MHz, vs 0.05-0.07 with tapers).
    """
    if shape_db is None:
        h = np.zeros(taps, dtype=np.complex128)
        h[taps // 2] = 1.0
        return h
    n = len(shape_db)
    h_full = sfft.ifft(sfft.ifftshift(10 ** (np.asarray(shape_db, dtype=np.float64) / 20)))
    k = np.arange(-(taps // 2), taps // 2 + 1)
    return np.asarray(h_full[k % n], dtype=np.complex128)


def fir_response(h: np.ndarray, m: int) -> np.ndarray:
    """|H| of a centred zero-phase FIR on an m-point bin grid (natural FFT order), float64."""
    arr = np.zeros(m, dtype=np.complex128)
    k = np.arange(-(len(h) // 2), len(h) // 2 + 1)
    arr[k % m] = h
    return np.abs(sfft.fft(arr))


class Frontend:
    """Noise generation, receiver response and ADC for one scene (sample rate fs, hop N, synthesis FFT M)."""

    def __init__(self, spec: NoiseSpec, t_sys_k: float, fs: int, hop: int, m: int, rng: KeyedRng) -> None:
        self.spec, self.fs, self.hop, self.m, self.rng = spec, fs, hop, m, rng
        h = design_fir(spec.shape_db)
        self.flat = spec.shape_db is None
        # Receiver response for signals on the synthesiser grid (None when flat: nothing to multiply).
        self.h_rx: np.ndarray | None = None if self.flat else fir_response(h, m).astype(np.float32)
        # Noise: white I/Q with unit variance each, scaled so that 0 dB of the shape is k T_sys (in mW).
        p_flat_mw = K_BOLTZ * t_sys_k * fs * 1000.0
        scale = np.sqrt(p_flat_mw / 2)
        self.taps = (h * scale).astype(np.complex64)
        self.n_hist = len(h) - 1
        self.ols_size = sfft.next_fast_len(hop + self.n_hist)
        self.taps_f = sfft.fft(self.taps, self.ols_size)
        # Per-component noise sigma (sqrt(mW)) after the FIR, and the ADC gain that maps it to sigma_counts.
        self.noise_sigma = float(scale * np.sqrt(np.sum(np.abs(h) ** 2)))
        self.gain = spec.sigma_counts / self.noise_sigma
        self._white: dict[int, np.ndarray] = {}

    def white(self, h: int) -> np.ndarray:
        """Unit-variance (per component) complex white noise of hop h, keyed ("noise", h). Cached for the next hop."""
        w = self._white.get(h)
        if w is None:
            g = self.rng.stream("noise", h)
            w = g.standard_normal(2 * self.hop, dtype=np.float32).view(np.complex64)
            self._white[h] = w
            for old in [k for k in self._white if k < h - 1 or k > h + 1]:
                del self._white[old]
        return w

    def noise(self, h: int) -> np.ndarray:
        """Shaped noise of hop h (complex64, sqrt(mW)), by overlap-save from hops h-1 and h."""
        seg = np.zeros(self.ols_size, dtype=np.complex64)
        seg[: self.n_hist] = self.white(h - 1)[self.hop - self.n_hist :]
        seg[self.n_hist : self.n_hist + self.hop] = self.white(h)
        y = sfft.ifft(sfft.fft(seg) * self.taps_f)
        return np.asarray(y[self.n_hist : self.n_hist + self.hop], dtype=np.complex64)

    def adc(self, y: np.ndarray) -> np.ndarray:
        """Interleaved uint8 I, Q: clip(rint(127.5 + G * y), 0, 255)."""
        v = np.ascontiguousarray(y, dtype=np.complex64).view(np.float32) * np.float32(self.gain)
        v += np.float32(127.5)
        np.rint(v, out=v)
        np.clip(v, 0, 255, out=v)
        return v.astype(np.uint8)
