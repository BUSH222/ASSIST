"""Twin of cpp/synth.cpp: overlap-add synthesis in the frequency domain."""

from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import scipy.fft as sfft

from assist.dsp_ref.basic import output

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray


@cache
def _hann(n: int) -> np.ndarray:
    """Periodic Hann window, computed in float64 and rounded to float32 (as the C++ kernel)."""
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)


def _add_scaled(acc: np.ndarray, src: np.ndarray, first: int, scale: np.float32, h: np.ndarray | None) -> None:
    """acc[k] += src[j] * scale * h[k] for k = (first + j) mod m."""
    m = len(acc)
    k = (first + np.arange(len(src))) % m
    v = src * scale
    if h is not None:
        v = v * h[k]
    acc[k] += v  # bins are distinct (len(src) <= m)


def synth(
    streams: Sequence[ArrayLike],
    lens: Sequence[int],
    bins: Sequence[int],
    active: Sequence[ArrayLike],
    b0: int,
    nb: int,
    m: int,
    h_rx: ArrayLike | None = None,
    out: np.ndarray | None = None,
) -> NDArray[np.complex64]:
    """Overlap-add synthesis of blocks b0 .. b0 + nb - 1 into (nb + 1) * m / 2 samples.

    Block b0 + i of emitter e is streams[e][i * L / 2 : i * L / 2 + L] (L = lens[e]); it is synthesised only where
    active[e][i] is set: V = FFT_L(x * hann_L) * (m / L) * (-1)^(k b) * h_rx, added at bins k - L/2 .. k + L/2 - 1
    (mod m), then y = IFFT_m(A), added at output offset i * m / 2.
    """
    ne = len(streams)
    if not (len(lens) == len(bins) == len(active) == ne):
        raise ValueError("streams, lens, bins and active must have the same length")
    if nb < 1:
        raise ValueError("nb must be >= 1")
    if m < 2 or m % 2:
        raise ValueError("m must be even and >= 2")
    xs, acts = [], []
    for e in range(ne):
        x = np.ascontiguousarray(streams[e], dtype=np.complex64)
        a = np.ascontiguousarray(active[e], dtype=np.uint8)
        if x.ndim != 1:
            raise ValueError("streams[i] must be 1-D")
        if a.ndim != 1:
            raise ValueError("active[i] must be 1-D")
        n = int(lens[e])
        if n < 2 or n % 2 or n > m:
            raise ValueError("lens[i] must be even, >= 2 and <= m")
        if len(x) < (nb + 1) * (n // 2):
            raise ValueError("streams[i] must hold (nb + 1) * lens[i] / 2 samples")
        if len(a) < nb:
            raise ValueError("active[i] must hold nb flags")
        xs.append(x)
        acts.append(a)
    h = None
    if h_rx is not None:
        h = np.ascontiguousarray(h_rx, dtype=np.float32)
        if h.ndim != 1:
            raise ValueError("h_rx must be 1-D")
        if len(h) != m:
            raise ValueError("h_rx must hold m values")
    hop = m // 2
    res = output(out, (nb + 1) * hop, np.complex64)
    res[:] = 0
    for i in range(nb):
        b = b0 + i
        acc: np.ndarray | None = None
        for e in range(ne):
            if not acts[e][i]:
                continue
            if acc is None:
                acc = np.zeros(m, dtype=np.complex64)
            n = int(lens[e])
            half = n // 2
            v = xs[e][i * half : i * half + n] * _hann(n)
            spec = sfft.fft(v)
            odd = (int(bins[e]) & 1) and (b & 1)
            scale = np.float32(m / n) * np.float32(-1.0 if odd else 1.0)
            k = int(bins[e])
            _add_scaled(acc, spec[half:], k - half, scale, h)
            _add_scaled(acc, spec[:half], k, scale, h)
        if acc is None:
            continue
        res[i * hop : i * hop + m] += sfft.ifft(acc)
    return res
