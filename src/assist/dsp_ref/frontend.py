"""Twins of cpp/frontend.cpp: white noise, noise shaping, ADC (plan §4.3, §4.4).

`white` reproduces the C++ kernel bit for bit: Philox4x32-10 in uint64 arithmetic, and a Box-Muller transform whose
ln, sin and cos are polynomials in float32 using only + - * / and sqrt, all exactly rounded, evaluated in the same
order as the C++ code (which is built without FMA contraction).
"""

from typing import TYPE_CHECKING

import numpy as np
import scipy.fft as sfft

from assist.dsp_ref.basic import output

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

_M32 = np.uint64(0xFFFFFFFF)
_S32 = np.uint64(32)
_F = np.float32


def _hx(s: str) -> np.float32:
    return np.float32(float.fromhex(s))


# Series coefficients, as hex floats in the C++ code.
_LN = [_hx("0x1.555556p-2"), _hx("0x1.99999ap-3"), _hx("0x1.24924ap-3"), _hx("0x1.c71c72p-4")]
_LN2_HI, _LN2_LO = _hx("0x1.62e4p-1"), _hx("0x1.7f7d1cp-20")
_SIN = [_hx("-0x1.555556p-3"), _hx("0x1.111112p-7"), _hx("-0x1.a01a02p-13"), _hx("0x1.71de3ap-19")]
_COS = [_hx("0x1.555556p-5"), _hx("-0x1.6c16c2p-10"), _hx("0x1.a01a02p-16"), _hx("0x1.27e4fcp-22")]
_SQRT2 = _F(1.41421356)
_HALF_PI = _F(1.57079633)


def _philox(c0: np.ndarray, c2: np.ndarray, c3: np.ndarray, key: int) -> np.ndarray:
    """Philox4x32-10 on counters (c0, 0, c2, c3) (uint64 arrays holding 32-bit values): words, shape (n, 4)."""
    c1 = np.zeros_like(c0)
    k0, k1 = np.uint64(key & 0xFFFFFFFF), np.uint64(key >> 32)
    for _ in range(10):
        p0 = np.uint64(0xD2511F53) * c0
        p1 = np.uint64(0xCD9E8D57) * c2
        c0, c1, c2, c3 = (p1 >> _S32) ^ c1 ^ k0, p1 & _M32, (p0 >> _S32) ^ c3 ^ k1, p0 & _M32
        k0 = (k0 + np.uint64(0x9E3779B9)) & _M32
        k1 = (k1 + np.uint64(0xBB67AE85)) & _M32
    return np.stack([c0, c1, c2, c3], axis=1).astype(np.uint32)


def _ln_poly(x: np.ndarray) -> np.ndarray:
    b = x.view(np.uint32)
    e = ((b >> 23) & 0xFF).astype(np.int32) - 127
    m = ((b & 0x7FFFFF) | 0x3F800000).astype(np.uint32).view(np.float32)
    big = m > _SQRT2
    m = np.where(big, m * _F(0.5), m)
    e = np.where(big, e + 1, e)
    s = (m - _F(1)) / (m + _F(1))
    s2 = s * s
    p = _F(2) * s * (_F(1) + s2 * (_LN[0] + s2 * (_LN[1] + s2 * (_LN[2] + s2 * _LN[3]))))
    fe = e.astype(np.float32)
    return fe * _LN2_HI + (fe * _LN2_LO + p)


def _box_muller(wa: np.ndarray, wb: np.ndarray) -> np.ndarray:
    u1 = (wa.astype(np.float32) + _F(0.5)) * _hx("0x1p-32")
    r = np.sqrt(_F(-2) * _ln_poly(u1))
    t = (((wb >> 6) & 0xFFFFFF).astype(np.float32) * _hx("0x1p-24") - _F(0.5)) * _HALF_PI
    t2 = t * t
    sn = t * (_F(1) + t2 * (_SIN[0] + t2 * (_SIN[1] + t2 * (_SIN[2] + t2 * _SIN[3]))))
    cs = _F(1) + t2 * (_F(-0.5) + t2 * (_COS[0] + t2 * (_COS[1] + t2 * (_COS[2] - t2 * _COS[3]))))
    q = wb >> 30
    x = np.select([q == 0, q == 1, q == 2], [cs, -sn, -cs], sn)
    y = np.select([q == 0, q == 1, q == 2], [sn, cs, -sn], -cs)
    out = np.empty(2 * len(wa), dtype=np.float32)
    out[0::2] = r * x
    out[1::2] = r * y
    return out.view(np.complex64)


def _white_range(key: int, hops: np.ndarray, i0: int, i1: int) -> np.ndarray:
    """Samples [i0, i1) (even) of each hop in `hops`, shape (len(hops), i1 - i0), complex64."""
    j = np.arange(i0 // 2, i1 // 2, dtype=np.uint64)
    h = np.asarray(hops, dtype=np.uint64)[:, None]
    c0 = np.broadcast_to(j, (len(hops), len(j))).ravel()
    c2 = np.broadcast_to(h & _M32, (len(hops), len(j))).ravel()
    c3 = np.broadcast_to(h >> _S32, (len(hops), len(j))).ravel()
    w = _philox(c0, c2, c3, key).ravel()  # 4 words per counter: samples 2j (w0, w1) and 2j + 1 (w2, w3)
    return _box_muller(w[0::2], w[1::2]).reshape(len(hops), i1 - i0)


def white(key: int, h0: int, nhops: int, hop: int) -> NDArray[np.complex64]:
    """Complex white Gaussian noise (unit variance per component) of hops h0 .. h0 + nhops - 1."""
    if nhops < 0:
        raise ValueError("nhops must be >= 0")
    if hop < 2 or hop % 2:
        raise ValueError("hop must be even and >= 2")
    if nhops == 0:
        return np.zeros(0, dtype=np.complex64)
    return _white_range(key, np.arange(h0, h0 + nhops, dtype=np.uint64), 0, hop).ravel()


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
    tf = np.ascontiguousarray(taps_f, dtype=np.complex64)
    if tf.ndim != 1:
        raise ValueError("taps_f must be 1-D")
    if h0 < 1:
        raise ValueError("h0 must be >= 1 (hop h0 - 1 supplies the history)")
    if nhops < 0:
        raise ValueError("nhops must be >= 0")
    if hop < 2 or hop % 2:
        raise ValueError("hop must be even and >= 2")
    if sub < 2 or sub % 2 or hop % sub:
        raise ValueError("sub must be even and divide hop")
    if n_hist < 0 or n_hist % 2 or n_hist > hop:
        raise ValueError("n_hist must be even and in [0, hop]")
    ols = len(tf)
    if ols < sub + n_hist:
        raise ValueError("taps_f must hold at least sub + n_hist bins")
    res = output(out, nhops * hop, np.complex64)
    if nhops == 0:
        return res
    hist = _white_range(key, np.array([h0 - 1], dtype=np.uint64), hop - n_hist, hop).ravel()
    wn = np.concatenate([hist, white(key, h0, nhops, hop)])  # contiguous white samples from n_hist before hop h0
    nseg = nhops * (hop // sub)
    rows = np.zeros((nseg, ols), dtype=np.complex64)
    idx = np.arange(nseg)[:, None] * sub + np.arange(n_hist + sub)
    rows[:, : n_hist + sub] = wn[idx]
    y = sfft.ifft(sfft.fft(rows, axis=1) * tf, axis=1)
    res[:] = y[:, n_hist : n_hist + sub].ravel()
    return res


def adc(x: ArrayLike, gain: float, add: ArrayLike | None = None, out: np.ndarray | None = None) -> NDArray[np.uint8]:
    """Interleaved uint8 I, Q: clip(rint(127.5 + gain * (x + add)), 0, 255); rint rounds half to even."""
    xa = np.ascontiguousarray(x, dtype=np.complex64)
    if xa.ndim != 1:
        raise ValueError("x must be 1-D")
    if add is not None:
        aa = np.ascontiguousarray(add, dtype=np.complex64)
        if aa.ndim != 1:
            raise ValueError("add must be 1-D")
        if len(aa) != len(xa):
            raise ValueError("add must have the same length as x")
        xa = xa + aa
    v = xa.view(np.float32) * np.float32(gain)
    v += np.float32(127.5)
    np.rint(v, out=v)
    np.clip(v, 0, 255, out=v)
    res = output(out, len(v), np.uint8)
    res[:] = v
    return res
