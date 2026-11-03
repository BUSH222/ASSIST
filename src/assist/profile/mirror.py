"""P9: mirror (image) measurements.

Image rejection at an offset: a strong source band [f_src +- half_bw] and its mirror about fc (bins k and
FFT_SIZE - k). For every spectrum (row or per-minute mean) the excess power over the local noise is measured in both
bands; the image ratio is sum(image excess) / sum(source excess) over the spectra where the source is strongest.

138.0-138.2 MHz events: for each event, the mirror band (2 fc - f, i.e. 135.8-136.0 MHz) is compared at the same
time. A stronger mirror means the event is the image of an in-window source; a mirror weaker by about the image
rejection means the event is real (or an alias from outside the window, which gives the same signature).
"""

from typing import TYPE_CHECKING

import numpy as np

from assist import waterfall as wf

if TYPE_CHECKING:
    from collections.abc import Callable

N = wf.FFT_SIZE


def mirror_bins(k: np.ndarray) -> np.ndarray:
    return (N - np.asarray(k)) % N


def band_excess(spectra_db: np.ndarray, bins: np.ndarray, guard: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """(linear excess power in `bins`, noise power in `bins`) per spectrum; noise from `guard` bins each side."""
    lo, hi = int(bins.min()), int(bins.max()) + 1
    side = np.r_[max(0, lo - guard) : lo, hi : min(N, hi + guard)]
    p = 10 ** (spectra_db[:, bins] / 10)
    n = np.median(10 ** (spectra_db[:, side] / 10), axis=1)
    return (p - n[:, None]).sum(axis=1), n * len(bins)


def image_rejection(
    spectra_db: np.ndarray, f_hz: np.ndarray, f_src_hz: float, half_bw_hz: float, top_fraction: float = 0.2
) -> dict[str, float]:
    """Image level relative to the source (dB) at offset f_src - fc, from the strongest `top_fraction` spectra."""
    bins = np.flatnonzero(np.abs(f_hz - f_src_hz) <= half_bw_hz)
    s, s_noise = band_excess(spectra_db, bins)
    i, _i_noise = band_excess(spectra_db, mirror_bins(bins)[::-1])
    k = max(5, int(len(s) * top_fraction))
    top = np.argsort(s)[-k:]
    ratio = float(i[top].sum() / s[top].sum())
    snr = float(np.median(s[top] / s_noise[top]))
    # Detection limit: the image excess is noise-limited at ~ (noise per band) / sqrt(n averages)
    floor = float(np.std(i[np.argsort(s)[: max(5, len(s) // 2)]]) / np.sqrt(k) / np.median(s[top]))
    return {
        "source_hz": float(f_src_hz),
        "image_db": float(10 * np.log10(ratio)) if ratio > 0 else float("-inf"),
        "detection_limit_db": float(10 * np.log10(max(floor, 1e-12))),
        "source_snr_db": float(10 * np.log10(snr)) if snr > 0 else float("-inf"),
        "spectra_used": k,
    }


def coherent_image(
    read_iq: Callable[[int, int], np.ndarray], n0: int, n1: int, fs: float, offset_hz: float, decim: int = 48
) -> dict[str, float]:
    """IQ-image test between fc + offset and fc - offset: fit mirror(t) = beta * conj(source(t)).

    A true image is coherent with the conjugate source (coherence -> its SNR-limited value); independent activity
    in the mirror band gives coherence ~0 even if its average level tracks the source.
    """
    from assist.profile.dsp import channelize

    s = channelize(read_iq, n0, n1, fs, offset_hz, decim)
    m = channelize(read_iq, n0, n1, fs, -offset_hz, decim)
    sc = np.conj(s).astype(np.complex128)
    m = m.astype(np.complex128)
    beta = np.vdot(sc, m) / np.vdot(sc, sc)
    coh = abs(np.vdot(sc, m)) ** 2 / (np.vdot(sc, sc).real * np.vdot(m, m).real)
    return {
        "offset_hz": float(offset_hz),
        "beta_db": float(10 * np.log10(abs(beta) ** 2 + 1e-30)),
        "beta_phase_deg": float(np.degrees(np.angle(beta))),
        "coherence": float(coh),
        "seconds": (n1 - n0) / fs,
    }
