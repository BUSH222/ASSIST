"""Twins of cpp/conv.cpp: K = 7, r = 1/2 convolutional encoder and the CCSDS randomiser (exact, integer maths)."""

from functools import cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

_POLY_A = 0x4F  # 171 octal, bit-reversed (newest bit in bit 0)
_POLY_B = 0x6D  # 133 octal


def _u8(x: ArrayLike, name: str) -> np.ndarray:
    a = np.ascontiguousarray(x, dtype=np.uint8)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    return a


@cache
def _parity() -> np.ndarray:
    v = np.arange(128)
    p = np.zeros(128, dtype=np.uint8)
    for b in range(7):
        p ^= ((v >> b) & 1).astype(np.uint8)
    return p


def conv_encode(bits: ArrayLike, state: int = 0) -> NDArray[np.uint8]:
    """K = 7, r = 1/2 convolutional code (171, 133 octal) of 0/1 bits; state = previous 6 input bits."""
    b = _u8(bits, "bits") & 1
    if not 0 <= state < 64:
        raise ValueError("state must hold 6 bits (0 .. 63)")
    n = len(b)
    # Register after bit i: bits i-6 .. i, the newest in bit 0 (earlier state bits fill in before the frame).
    hist = np.concatenate([np.array([(state >> (5 - j)) & 1 for j in range(6)], dtype=np.int64), b.astype(np.int64)])
    sr = np.zeros(n, dtype=np.int64)
    for d in range(7):  # bit d of the register = the input d steps back
        sr |= hist[6 - d : 6 - d + n] << d
    p = _parity()
    out = np.empty(2 * n, dtype=np.uint8)
    out[0::2] = p[sr & _POLY_A]
    out[1::2] = p[sr & _POLY_B]
    return out


@cache
def _pn(n: int) -> np.ndarray:
    """The first n bytes of the CCSDS randomiser sequence (b[k + 8] = b[k + 7] ^ b[k + 5] ^ b[k + 3] ^ b[k])."""
    bits = np.ones(8 * n + 8, dtype=np.uint8)
    for k in range(8 * n):
        bits[k + 8] = bits[k + 7] ^ bits[k + 5] ^ bits[k + 3] ^ bits[k]
    return np.packbits(bits[: 8 * n])


def ccsds_randomise(data: ArrayLike) -> NDArray[np.uint8]:
    """XOR with the CCSDS randomiser sequence (FF 48 0E ...)."""
    d = _u8(data, "data")
    return d ^ _pn(len(d))
