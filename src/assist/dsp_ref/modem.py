"""Twins of cpp/modem.cpp: random payload bytes, pulse shaping, LoRa chirps, carrier NCO.

`random_bytes` is exact (integer maths). The others compute in float64 like the C++ code, in a different order, so
they agree to rounding.
"""

from typing import TYPE_CHECKING

import numpy as np

from assist.dsp_ref.basic import output
from assist.dsp_ref.frontend import _philox

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def random_bytes(key: int, block: int, n: int) -> NDArray[np.uint8]:
    """n random bytes of block `block` (Philox4x32-10, counter (i, 0, block))."""
    if n < 0:
        raise ValueError("n must be >= 0")
    m = (n + 15) // 16
    c0 = np.arange(m, dtype=np.uint64)
    c2 = np.full(m, block & 0xFFFFFFFF, dtype=np.uint64)
    c3 = np.full(m, block >> 32, dtype=np.uint64)
    words = _philox(c0, c2, c3, key)  # (m, 4) uint32
    return np.ascontiguousarray(words.astype("<u4")).view(np.uint8).reshape(-1)[:n].copy()


def _f32(x: ArrayLike, name: str) -> np.ndarray:
    a = np.ascontiguousarray(x, dtype=np.float32)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    return a


def _branch(s: np.ndarray, k0: int, h: np.ndarray, sps: int, m0: int, n: int) -> np.ndarray:
    """sum_k s[k - k0] h[m - k sps + c] for m = m0 .. m0 + n - 1 (float64)."""
    c = (len(h) - 1) // 2
    up = np.zeros(len(s) * sps, dtype=np.float64)
    up[::sps] = s  # sample k0 sps + i sps holds symbol k0 + i
    y = np.convolve(up, h.astype(np.float64))  # y[q] is at sample k0 sps + q - c
    q = np.arange(m0, m0 + n, dtype=np.int64) - k0 * sps + c
    ok = (q >= 0) & (q < len(y))
    out = np.zeros(n, dtype=np.float64)
    out[ok] = y[q[ok]]
    return out


def shape(
    si: ArrayLike, sq: ArrayLike, k0: int, h: ArrayLike, sps: int, qd: int, n0: int, n: int
) -> NDArray[np.complex64]:
    """Pulse shaping of symbols k0 .. into samples n0 .. n0 + n - 1."""
    a, b, t = _f32(si, "si"), _f32(sq, "sq"), _f32(h, "h")
    if len(a) != len(b):
        raise ValueError("si and sq must have the same length")
    if len(t) < 1 or len(t) % 2 != 1:
        raise ValueError("h must hold an odd number of taps")
    if sps < 1:
        raise ValueError("sps must be >= 1")
    if not 0 <= qd < sps:
        raise ValueError("qd must be in [0, sps)")
    if n < 0:
        raise ValueError("n must be >= 0")
    if len(a) == 0:
        return np.zeros(n, dtype=np.complex64)
    re = _branch(a, k0, t, sps, n0, n)
    im = _branch(b, k0, t, sps, n0 - qd, n)
    out = np.empty(n, dtype=np.complex64)
    out.real = re.astype(np.float32)
    out.imag = im.astype(np.float32)
    return out


def chirp(
    start: ArrayLike, length: ArrayLike, val: ArrayLike, sf: int, bw: float, t0: float, dt: float, n: int
) -> NDArray[np.complex64]:
    """A LoRa burst from chirp segments, sampled at t0 + j dt."""
    st = np.ascontiguousarray(start, dtype=np.float64)
    ln = np.ascontiguousarray(length, dtype=np.float64)
    v = np.ascontiguousarray(val, dtype=np.int32)
    if st.ndim != 1 or ln.ndim != 1 or v.ndim != 1:
        raise ValueError("start, length and val must be 1-D")
    if not (st.shape == ln.shape == v.shape):
        raise ValueError("start, length and val must match")
    if st.size < 1:
        raise ValueError("at least one segment is needed")
    if not 5 <= sf <= 12:
        raise ValueError("sf must be in 5 .. 12")
    if bw <= 0 or dt <= 0:
        raise ValueError("bw and dt must be > 0")
    if n < 0:
        raise ValueError("n must be >= 0")
    if np.any(ln <= 0):
        raise ValueError("segment lengths must be > 0")
    if st.size > 1 and np.any(st[1:] <= st[:-1]):
        raise ValueError("segment starts must increase")
    if np.any(v >= (1 << sf)):
        raise ValueError("chirp values must be < 2^sf")
    nn = float(2**sf)
    t_sym = nn / bw
    slope = bw / t_sym
    tau = t0 + np.arange(n, dtype=np.float64) * dt
    seg = np.searchsorted(st + ln, tau, side="right")  # first segment whose end is after tau
    inside = (seg < st.size) & (tau >= st[np.minimum(seg, st.size - 1)])
    out = np.zeros(n, dtype=np.complex64)
    s = seg[inside]
    u = tau[inside] - st[s]
    vs = v[s].astype(np.float64)
    up = v[s] >= 0
    t_wrap = t_sym * (1.0 - vs / nn)
    cyc_up = (vs * bw / nn - 0.5 * bw) * u + 0.5 * slope * u * u - np.where(u > t_wrap, bw * (u - t_wrap), 0.0)
    cyc_dn = 0.5 * bw * u - 0.5 * slope * u * u
    cyc = np.where(up, cyc_up, cyc_dn)
    out[inside] = np.exp(2j * np.pi * (cyc - np.floor(cyc))).astype(np.complex64)
    return out


def nco(x: ArrayLike | None, gain: ArrayLike, cyc: ArrayLike, out: np.ndarray | None = None) -> NDArray[np.complex64]:
    """x * gain * exp(2 pi i cyc) as complex64 (x None: a constant 1)."""
    g = np.ascontiguousarray(gain, dtype=np.float64)
    c = np.ascontiguousarray(cyc, dtype=np.float64)
    if g.ndim != 1 or c.ndim != 1:
        raise ValueError("gain and cyc must be 1-D")
    if len(c) != len(g):
        raise ValueError("gain and cyc must have the same length")
    v = g * np.exp(2j * np.pi * (c - np.floor(c)))
    if x is not None:
        xa = np.ascontiguousarray(x, dtype=np.complex64)
        if xa.ndim != 1:
            raise ValueError("x must be 1-D")
        if len(xa) != len(g):
            raise ValueError("x must have the same length as gain")
        v = v * xa.astype(np.complex128)
    res = output(out, len(g), np.complex64)
    res[:] = v
    return res
