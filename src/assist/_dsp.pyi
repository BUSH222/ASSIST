"""Type stubs for the C++ module assist._dsp (cpp/bindings.cpp). Keep in sync with assist.dsp_ref.

Kernels with `out` write into it (exact dtype, 1-D, C-contiguous, writable, the right length) and return it.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

_native: bool  # built with -march=native (ASSIST_NATIVE=ON)

def scale(x: ArrayLike, g: complex) -> NDArray[np.complex64]:
    """x * g as complex64 (x: 1-D, cast to complex64)."""

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
    """Overlap-add synthesis of blocks b0 .. b0 + nb - 1 into (nb + 1) * m / 2 samples (plan §4.2)."""

def white(key: int, h0: int, nhops: int, hop: int) -> NDArray[np.complex64]:
    """Complex white Gaussian noise (unit variance per component) of hops h0 .. h0 + nhops - 1."""

def noise(
    key: int,
    h0: int,
    nhops: int,
    hop: int,
    sub: int,
    taps_f: ArrayLike,
    n_hist: int,
    out: np.ndarray | None = None,
) -> NDArray[np.complex64]:
    """Shaped noise of hops h0 .. h0 + nhops - 1 by overlap-save in segments of `sub` samples (plan §4.3)."""

def adc(x: ArrayLike, gain: float, add: ArrayLike | None = None, out: np.ndarray | None = None) -> NDArray[np.uint8]:
    """Interleaved uint8 I, Q: clip(rint(127.5 + gain * (x + add)), 0, 255)."""
