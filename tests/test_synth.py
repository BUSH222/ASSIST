"""T-OLA (plan §10.1): single signals through the overlap-add synthesiser.

The synthesiser is driven directly: an emitter at narrow rate r and coarse bin k gets its narrow samples
[bL/2, bL/2 + L) per block b, Hann-windowed, and the output hops are compared with the same signal computed
directly at fs. Block indices are absolute (around 1.8e11 in 2026), so the (-1)^(k b) hop phase is exercised.

- CW at several bins (odd and even) and offsets from the bin;
- QPSK at 72 ksym/s with an RRC pulse (alpha 0.6) at 4 samples per symbol, a stand-in for the M6 `oqpsk` signal:
  the narrow samples and the wideband reference come from the same continuous-time waveform.

Checks: error and spurious lines <= -60 dBc, CW amplitude flat within 0.01 dB, no phase jump at hop edges.
"""

import numpy as np
import pytest

from assist.engine.synth import add_block, hann, overlap, to_time

FS = 2_400_000
HOP = FS // 100
M = 2 * HOP
B0 = 179_379_411_740  # an absolute block index in 2026 (Unix time / 10 ms)


def _synth(narrow, rate: int, k_bin: int, h0: int, hops: int) -> np.ndarray:
    """Output hops h0 .. h0 + hops - 1, with narrow(n0, n) giving narrow samples at absolute indices."""
    n = rate * M // FS
    blocks = {}
    for b in range(h0 - 1, h0 + hops):
        acc = np.zeros(M, dtype=np.complex64)
        v = (narrow(b * (n // 2), n) * hann(n)).astype(np.complex64)
        add_block(acc, v, k_bin, b)
        blocks[b] = to_time(acc, None)
    return np.concatenate([overlap(blocks[h - 1], blocks[h], HOP) for h in range(h0, h0 + hops)])


def _tone(n_abs: np.ndarray, f_hz: int, rate: int) -> np.ndarray:
    """exp(j 2 pi f n / rate) with the phase reduced exactly (integer f, absolute integer n).

    Both factors are reduced mod rate first: f * n itself overflows int64 (4e15 * 3.5e5).
    """
    return np.exp(2j * np.pi * (((f_hz % rate) * (n_abs % rate)) % rate) / rate)


def _dbc(err: np.ndarray, ref: np.ndarray) -> float:
    return float(10 * np.log10(np.sum(np.abs(err) ** 2) / np.sum(np.abs(ref) ** 2)))


def _max_line_dbc(err: np.ndarray, ref: np.ndarray) -> float:
    """Strongest spectral line of the error relative to the signal's total power (Blackman-Harris window)."""
    from scipy.signal.windows import blackmanharris

    w = blackmanharris(len(err))
    line = np.max(np.abs(np.fft.fft(err * w)) ** 2) / np.sum(w) ** 2  # power of the strongest line
    return float(10 * np.log10(line / np.mean(np.abs(ref) ** 2)))


@pytest.mark.parametrize(
    ("k_bin", "delta_hz", "rate"),
    [(7000, 0, 11_000), (7001, 1234, 11_000), (-12345, -4321, 12_000), (23_999, 2_500, 12_000), (-24_000, 17, 2000)],
)
def test_ola_cw(k_bin: int, delta_hz: int, rate: int) -> None:
    hops = 40
    h0 = B0 + 3
    y = _synth(lambda n0, n: _tone(n0 + np.arange(n, dtype=np.int64), delta_hz, rate), rate, k_bin, h0, hops)
    n_abs = h0 * HOP + np.arange(hops * HOP, dtype=np.int64)
    ideal = _tone(n_abs, k_bin * 50 + delta_hz, FS)
    err = y.astype(np.complex128) - ideal
    assert _dbc(err, ideal) <= -60
    assert _max_line_dbc(err, ideal) <= -60
    amp_db = 20 * np.log10(np.abs(y))
    assert np.max(np.abs(amp_db)) <= 0.01  # flat across hops
    ph = np.angle(y.astype(np.complex128) * np.conj(ideal))
    assert np.max(np.abs(ph)) < 1e-3
    edges = np.arange(1, hops) * HOP
    assert np.max(np.abs(ph[edges] - ph[edges - 1])) < 1e-4  # no phase step at hop boundaries


SYM_RATE = 72_000
QPSK_RATE = 4 * SYM_RATE  # 288 kHz narrow rate
SPAN = 12  # symbols each side
ALPHA = 0.6


def _rrc(t: np.ndarray) -> np.ndarray:
    """RRC pulse (alpha 0.6, unit symbol time) at times t in symbols, Kaiser-tapered over +-SPAN symbols."""
    a = ALPHA
    t = np.asarray(t, dtype=np.float64)
    out = np.empty_like(t)
    zero = np.isclose(t, 0.0, atol=1e-12)
    sing = np.isclose(np.abs(t), 1 / (4 * a), atol=1e-12)
    reg = ~(zero | sing)
    tr = t[reg]
    out[reg] = (np.sin(np.pi * tr * (1 - a)) + 4 * a * tr * np.cos(np.pi * tr * (1 + a))) / (
        np.pi * tr * (1 - (4 * a * tr) ** 2)
    )
    out[zero] = 1 - a + 4 * a / np.pi
    out[sing] = (
        a / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * a)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * a)))
    )
    taper = np.kaiser(2001, 8.0)
    idx = np.clip(np.round((t / SPAN + 1) * 1000).astype(int), 0, 2000)
    return np.where(np.abs(t) <= SPAN, out * taper[idx], 0.0)


class _Qpsk:
    """A continuous-time QPSK waveform: sum of s_k p(t - k T), symbols keyed by index, t from an origin t_ref."""

    def __init__(self, t_ref_s: float, n_sym: int, seed: int = 3) -> None:
        g = np.random.default_rng(seed)
        self.t_ref = t_ref_s
        self.s = (g.choice([-1.0, 1.0], n_sym) + 1j * g.choice([-1.0, 1.0], n_sym)) / np.sqrt(2)

    def at(self, t_s: np.ndarray) -> np.ndarray:
        u = (t_s - self.t_ref) * SYM_RATE  # time in symbols from the origin
        k0 = np.floor(u).astype(int)
        out = np.zeros(u.shape, dtype=np.complex128)
        for d in range(-SPAN, SPAN + 2):
            k = k0 + d
            ok = (k >= 0) & (k < len(self.s))
            out[ok] += self.s[k[ok]] * _rrc(u[ok] - k[ok])
        return out


def test_ola_qpsk() -> None:
    """QPSK (stand-in for oqpsk until M6), offset 3 kHz from an odd bin: error <= -60 dBc against the direct signal."""
    hops = 12
    h0 = B0 + 7
    k_bin, delta = 7001, 3000
    t_ref = (h0 - 2) * HOP / FS  # origin a little before the first block; times below are relative to it
    q = _Qpsk(0.0, int((hops + 4) * HOP / FS * SYM_RATE) + 4 * SPAN)

    def narrow(n0: int, n: int) -> np.ndarray:
        n_abs = n0 + np.arange(n, dtype=np.int64)
        t = (n_abs - (h0 - 2) * HOP * QPSK_RATE // FS) / QPSK_RATE  # exact: (h0 - 2) * HOP * r / fs is an integer
        return q.at(t + 0.0) * _tone(n_abs, delta, QPSK_RATE)

    y = _synth(narrow, QPSK_RATE, k_bin, h0, hops)
    n_abs = h0 * HOP + np.arange(hops * HOP, dtype=np.int64)
    ref = q.at((n_abs - (h0 - 2) * HOP) / FS) * _tone(n_abs, k_bin * 50 + delta, FS)
    assert t_ref > 0
    err = y.astype(np.complex128) - ref
    assert _dbc(err, ref) <= -60
    assert _max_line_dbc(err, ref) <= -60


def test_hann_sums_to_one() -> None:
    for n in (200, 216, 5760):
        w = hann(n)
        np.testing.assert_allclose(w[: n // 2] + w[n // 2 :], 1.0, atol=1e-15)


def test_add_block_wraps_around_the_spectrum() -> None:
    """A block near the edge of the M-bin grid wraps (mod M), as an RTL alias would."""
    acc = np.zeros(M, dtype=np.complex64)
    v = np.ones(200, dtype=np.complex64)  # DC only
    add_block(acc, v, M // 2 - 10, 0)
    assert np.argmax(np.abs(acc)) == M // 2 - 10
    acc2 = np.zeros(M, dtype=np.complex64)
    add_block(acc2, v * np.exp(-2j * np.pi * 50 * np.arange(200) / 10_000).astype(np.complex64), 0, 0)
    assert np.argmax(np.abs(acc2)) == M - 1  # bin -1
