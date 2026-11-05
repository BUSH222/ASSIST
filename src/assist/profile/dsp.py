"""Signal-processing helpers for the profile tool (numpy/scipy; not simulator kernels)."""

from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import firwin, upfirdn

if TYPE_CHECKING:
    from collections.abc import Callable


def lowpass_taps(fs: float, decim: int, n_taps: int = 0) -> np.ndarray:
    """Low-pass FIR for decimation by `decim`: passband to 0.4 of the output rate, ~80 dB stopband."""
    if n_taps <= 0:
        n_taps = 16 * decim + 1
    return firwin(n_taps, 0.4 * fs / decim, fs=fs, window=("kaiser", 8.0)).astype(np.float64)


def channelize(
    read_iq: Callable[[int, int], np.ndarray],
    n0: int,
    n1: int,
    fs: float,
    f_offset_hz: float,
    decim: int,
    taps: np.ndarray | None = None,
    chunk: int = 2_400_000,
) -> np.ndarray:
    """Mix samples [n0, n1) down by f_offset_hz and decimate. Output sample m is input sample n0 + m*decim.

    The mixer phase is addressed by absolute sample index, so any window gives the same output for the same
    samples. Filtering uses the samples before n0 (when available), so chunk edges are seamless; the filter is
    zero-phase (output aligned to its input sample, group delay removed).
    """
    if taps is None:
        taps = lowpass_taps(fs, decim)
    n_t = len(taps)
    delay = (n_t - 1) // 2
    m_total = (n1 - n0 + decim - 1) // decim
    out = np.empty(m_total, dtype=np.complex64)
    w = 2 * np.pi * f_offset_hz / fs
    step = max(decim, chunk - chunk % decim)
    for m0 in range(0, m_total, step // decim):
        m1 = min(m_total, m0 + step // decim)
        # Output m (absolute input index c = n0 + m*decim) needs inputs c - delay .. c + delay.
        a = n0 + m0 * decim - delay
        b = n0 + (m1 - 1) * decim + delay + 1
        x = read_iq(max(a, 0), b - max(a, 0)).astype(np.complex128)
        if a < 0:
            x = np.concatenate([np.zeros(-a, np.complex128), x])
        if len(x) < b - a:
            x = np.concatenate([x, np.zeros(b - a - len(x), np.complex128)])
        idx = np.arange(a, b, dtype=np.float64)
        x *= np.exp(-1j * np.remainder(w * idx, 2 * np.pi))
        y = upfirdn(taps, x, up=1, down=decim)
        # upfirdn output j corresponds to input index a + j*decim - (n_t - 1) + ... ; with full convolution,
        # y[j] = sum_k h[k] x[j*decim - k]; the centred output for input position p = a + j*decim - delay.
        j0 = (2 * delay) // decim
        if (2 * delay) % decim:
            raise ValueError("filter length must give a group delay that is a multiple of decim/2")
        out[m0:m1] = y[j0 : j0 + (m1 - m0)]
    return out


def rc_psd(f: np.ndarray, rs: float, alpha: float) -> np.ndarray:
    """Raised-cosine spectrum shape (the PSD of an RRC-shaped signal), peak 1."""
    af = np.abs(f)
    f1 = (1 - alpha) * rs / 2
    f2 = (1 + alpha) * rs / 2
    out = np.zeros_like(af, dtype=np.float64)
    out[af <= f1] = 1.0
    mid = (af > f1) & (af <= f2)
    out[mid] = 0.5 * (1 + np.cos(np.pi / (alpha * rs) * (af[mid] - f1)))
    return out


def runs(on: np.ndarray) -> np.ndarray:
    """[start, end) index pairs of True runs, shape (k, 2)."""
    d = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
    return np.stack([np.flatnonzero(d == 1), np.flatnonzero(d == -1)], axis=1)


def hysteresis(x: np.ndarray, on_th: float, off_th: float) -> np.ndarray:
    """Boolean on/off series: switches on above on_th, off below off_th (on_th >= off_th)."""
    above = x > on_th
    keep = x > off_th
    out = np.zeros(len(x), dtype=bool)
    for s, e in runs(keep):
        if above[s:e].any():
            out[s:e] = True
    return out
