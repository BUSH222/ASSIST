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
    """Overlap-add synthesis of blocks b0 .. b0 + nb - 1 into (nb + 1) * m / 2 samples."""

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
    """Shaped noise of hops h0 .. h0 + nhops - 1 by overlap-save in segments of `sub` samples."""

def adc(x: ArrayLike, gain: float, add: ArrayLike | None = None, out: np.ndarray | None = None) -> NDArray[np.uint8]:
    """Interleaved uint8 I, Q: clip(rint(127.5 + gain * (x + add)), 0, 255)."""

def random_bytes(key: int, block: int, n: int) -> NDArray[np.uint8]:
    """n random bytes of block `block` (Philox4x32-10, counter (i, 0, block))."""

def conv_encode(bits: ArrayLike, state: int = 0) -> NDArray[np.uint8]:
    """K = 7, r = 1/2 convolutional code (171, 133 octal) of 0/1 bits; state = previous 6 input bits."""

def ccsds_randomise(data: ArrayLike) -> NDArray[np.uint8]:
    """XOR with the CCSDS randomiser sequence (FF 48 0E ...)."""

def shape(
    si: ArrayLike, sq: ArrayLike, k0: int, h: ArrayLike, sps: int, qd: int, n0: int, n: int
) -> NDArray[np.complex64]:
    """Pulse shaping of symbols k0 .. into samples n0 .. n0 + n - 1."""

def chirp(
    start: ArrayLike, length: ArrayLike, val: ArrayLike, sf: int, bw: float, t0: float, dt: float, n: int
) -> NDArray[np.complex64]:
    """A LoRa burst from chirp segments, sampled at t0 + j dt."""

def nco(x: ArrayLike | None, gain: ArrayLike, cyc: ArrayLike, out: np.ndarray | None = None) -> NDArray[np.complex64]:
    """x * gain * exp(2 pi i cyc) as complex64 (x None: a constant 1)."""
