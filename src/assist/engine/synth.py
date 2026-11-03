"""Wideband synthesis by overlap-add in the frequency domain (plan §4.2).

Hop N = fs / 100 (10 ms), FFT size M = 2N, bin spacing fs / M = 50 Hz. Block b covers the absolute wideband samples
[bN, bN + 2N). An emitter at narrow rate r has L = r M / fs samples per block (even, since r is a multiple of 100):
narrow samples [bL/2, bL/2 + L) cover exactly the same time span.

Per block and emitter: v = x * a * exp(j phi) * hann_periodic(L); V = FFT_L(v); V * (M / L) * (-1)^(k b) is added to
the accumulator at bins k - L/2 ... k + L/2 - 1 (mod M), where k is the emitter's coarse bin (f_bin = fc + k * 50 Hz).
The (-1)^(k b) factor keeps the coarse shift phase-continuous from hop to hop, referred to the Unix epoch. Then
A *= H_rx and y = IFFT_M(A); the periodic Hann at hop L/2 sums to one, so hop h of the output is
y_{h-1}[N:] + y_h[:N].
"""

from functools import cache

import numpy as np
import scipy.fft as sfft


@cache
def hann(n: int) -> np.ndarray:
    """Periodic Hann window (float64): w[j] + w[j + n/2] = 1."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def add_block(acc: np.ndarray, v: np.ndarray, k_bin: int, b: int) -> None:
    """Add a windowed narrow block v (L samples) into the M-bin accumulator, centred on bin k_bin, for block b."""
    m, n = len(acc), len(v)
    spec = sfft.fft(v.astype(np.complex64, copy=False))
    scale = np.float32(m / n) * np.float32(-1.0 if (k_bin * b) & 1 else 1.0)
    spec *= scale
    half = n // 2
    # spec[0:half] are bins 0 .. half-1, spec[half:] are bins -half .. -1.
    for src, first in ((spec[half:], k_bin - half), (spec[:half], k_bin)):
        start = first % m
        end = start + len(src)
        if end <= m:
            acc[start:end] += src
        else:
            cut = m - start
            acc[start:] += src[:cut]
            acc[: end - m] += src[cut:]


def to_time(acc: np.ndarray, h_rx: np.ndarray | None) -> np.ndarray:
    """Apply the receiver response and return the block in time (complex64, M samples)."""
    if h_rx is not None:
        acc *= h_rx
    return np.asarray(sfft.ifft(acc), dtype=np.complex64)


def overlap(prev: np.ndarray | None, cur: np.ndarray | None, hop: int) -> np.ndarray:
    """Output hop h from blocks h-1 and h (either may be None: silent)."""
    out = np.zeros(hop, dtype=np.complex64)
    if prev is not None:
        out += prev[hop:]
    if cur is not None:
        out += cur[:hop]
    return out
