"""Every C++ kernel in assist._dsp matches its numpy twin in assist.dsp_ref (rtol 1e-5)."""

import os
import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest

from assist import _dsp, dsp_ref

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

RTOL = 1e-5


def _random_iq(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


def test_twins_expose_the_same_kernels() -> None:
    native = {name for name in dir(_dsp) if not name.startswith("_")}
    assert native == set(dsp_ref.__all__)


@pytest.mark.parametrize("g", [1.0, 0.3 - 1.7j, -2.5e-3 + 4e2j])
def test_scale_matches_reference(g: complex) -> None:
    x = _random_iq(100_003, seed=1)
    got = _dsp.scale(x, g)
    want = dsp_ref.scale(x, g)
    assert got.dtype == want.dtype == np.complex64
    assert got.shape == want.shape == x.shape
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=0)


@pytest.mark.parametrize(
    "x",
    [
        _random_iq(1001, seed=2)[::3],  # non-contiguous
        _random_iq(64, seed=3).astype(np.complex128),  # cast down to complex64
        [1.0, 2j, -3.0 + 0.5j],  # plain list
        np.zeros(0, dtype=np.complex64),  # empty
    ],
    ids=["strided", "complex128", "list", "empty"],
)
def test_scale_input_conversion(x: ArrayLike) -> None:
    got = _dsp.scale(x, 2 - 1j)
    want = dsp_ref.scale(x, 2 - 1j)
    assert got.dtype == np.complex64
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=0)


@pytest.mark.parametrize("impl", [_dsp.scale, dsp_ref.scale], ids=["c++", "numpy"])
def test_scale_rejects_2d(impl) -> None:
    with pytest.raises(ValueError, match="1-D"):
        impl(np.zeros((2, 2), dtype=np.complex64), 1.0)


@pytest.mark.parametrize(("value", "expected"), [("", "c++"), ("0", "c++"), ("1", "numpy"), ("yes", "numpy")])
def test_assist_pure_selects_backend(value: str, expected: str) -> None:
    env = {**os.environ, "ASSIST_PURE": value}
    out = subprocess.run(
        [sys.executable, "-c", "from assist import backend; print(backend.NAME, backend.dsp.__name__)"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert out[0] == expected
    assert out[1] == ("assist.dsp_ref" if expected == "numpy" else "assist._dsp")


# --- FFT kernels: max |c++ - numpy| <= RTOL * peak (an element-wise rtol is undefined for bins near zero) ---


def _peak_err(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.max(np.abs(got.astype(np.complex128) - want)) / np.max(np.abs(want)))


M = 48_000
B0 = 179_379_411_740  # an absolute block index in 2026


def _synth_case(seed: int, nb: int = 7) -> tuple[list[np.ndarray], list[int], list[int], list[np.ndarray]]:
    """Emitters with several block lengths, bins of both signs, one near the wrap, mixed activity."""
    g = np.random.default_rng(seed)
    streams, lens, bins, active = [], [], [], []
    for n, k in [(220, 7001), (216, -12345), (5760, 23_990), (2, -24_000), (220, 7001)]:
        x = g.standard_normal((nb + 1) * n // 2) + 1j * g.standard_normal((nb + 1) * n // 2)
        streams.append(x.astype(np.complex64))
        lens.append(n)
        bins.append(k)
        active.append((g.random(nb) < 0.7).astype(np.uint8))
    active[0][:] = 1
    active[1][2] = 0  # block 2: only some emitters
    for a in active:
        a[4] = 0  # block 4: silent
    return streams, lens, bins, active


@pytest.mark.parametrize("h_rx", [None, "shape"], ids=["flat", "h_rx"])
@pytest.mark.parametrize("b0", [B0, B0 + 1, 0])
def test_synth_matches_reference(h_rx: str | None, b0: int) -> None:
    streams, lens, bins, active = _synth_case(5)
    h = None if h_rx is None else np.random.default_rng(6).uniform(0.3, 1.2, M).astype(np.float32)
    got = _dsp.synth(streams, lens, bins, active, b0, 7, M, h)
    want = dsp_ref.synth(streams, lens, bins, active, b0, 7, M, h)
    assert got.dtype == want.dtype == np.complex64
    assert got.shape == want.shape == (8 * M // 2,)
    assert _peak_err(got, want) <= RTOL
    hop = M // 2
    assert not np.any(got[4 * hop + hop : 5 * hop])  # silent block 4 adds nothing to the second half of its span
    np.testing.assert_array_equal(got[4 * hop + hop : 5 * hop], want[4 * hop + hop : 5 * hop])


def test_synth_writes_into_out() -> None:
    streams, lens, bins, active = _synth_case(7)
    for impl in (_dsp, dsp_ref):
        out = np.full(8 * M // 2, 99 + 0j, dtype=np.complex64)
        res = impl.synth(streams, lens, bins, active, B0, 7, M, None, out)
        assert res is out or np.shares_memory(res, out)
        np.testing.assert_array_equal(out, impl.synth(streams, lens, bins, active, B0, 7, M, None))


def test_synth_block_alone_equals_block_in_a_run() -> None:
    """A block synthesised on its own gives the same bits as inside a longer call (the engine's lead-in block)."""
    streams, lens, bins, active = _synth_case(8)
    full = _dsp.synth(streams, lens, bins, active, B0, 7, M)
    hop = M // 2
    i = 6  # the last block: its second half has no other contribution
    one = _dsp.synth(
        [x[i * n // 2 :] for x, n in zip(streams, lens, strict=True)],
        lens,
        bins,
        [a[i:] for a in active],
        B0 + i,
        1,
        M,
    )
    np.testing.assert_array_equal(one[hop:], full[(i + 1) * hop :])


def test_white_is_bit_identical() -> None:
    for key, h0 in [(0x0123456789ABCDEF, B0), (2**64 - 1, 2**40 + 3), (0, 1)]:
        got = _dsp.white(key, h0, 3, 24_000)
        want = dsp_ref.white(key, h0, 3, 24_000)
        assert got.dtype == want.dtype == np.complex64
        assert np.array_equal(got.view(np.uint64), want.view(np.uint64))


def test_white_is_random_access_and_keyed() -> None:
    run = _dsp.white(5, B0, 4, 2000)
    np.testing.assert_array_equal(_dsp.white(5, B0 + 2, 1, 2000), run[4000:6000])
    assert not np.array_equal(_dsp.white(6, B0 + 2, 1, 2000), run[4000:6000])  # another key
    assert not np.array_equal(run[:2000], run[2000:4000])  # another hop


def test_white_statistics() -> None:
    """Unit-variance complex Gaussian: moments, tails beyond the ADC clip level (4.78 sigma), I/Q independence."""
    from scipy import stats

    x = _dsp.white(0xA5A5A5A5DEADBEEF, B0, 100, 24_000).astype(np.complex128)
    v = np.concatenate([x.real, x.imag])
    n = len(v)
    assert abs(v.mean()) < 5 / np.sqrt(n)
    assert abs(v.var() - 1) < 5 * np.sqrt(2 / n)
    assert abs(stats.kurtosis(v)) < 5 * np.sqrt(24 / n)
    for t in (3.0, 4.0, 127.5 / 26.6868):
        expect = 2 * stats.norm.sf(t) * n
        assert abs(np.count_nonzero(np.abs(v) > t) - expect) < 5 * np.sqrt(expect) + 1
    assert abs(np.mean(x.real * x.imag)) < 5 / np.sqrt(len(x))
    assert abs(np.mean(x.real[1:] * x.real[:-1])) < 5 / np.sqrt(len(x))
    assert stats.kstest(v[:1_000_000], "norm").pvalue > 1e-3


def _taps_f(ols: int) -> np.ndarray:
    g = np.random.default_rng(9)
    taps = (g.standard_normal(513) + 1j * g.standard_normal(513)).astype(np.complex64)
    return np.asarray(np.fft.fft(taps, ols), dtype=np.complex64)


@pytest.mark.parametrize(("hop", "sub", "ols"), [(24_000, 8000, 8640), (24_000, 24_000, 24_576), (2000, 1000, 1600)])
def test_noise_matches_reference(hop: int, sub: int, ols: int) -> None:
    tf = _taps_f(ols)
    got = _dsp.noise(77, B0, 5, hop, sub, tf, 512)
    want = dsp_ref.noise(77, B0, 5, hop, sub, tf, 512)
    assert got.dtype == want.dtype == np.complex64
    assert got.shape == want.shape == (5 * hop,)
    assert _peak_err(got, want) <= RTOL


def test_noise_is_the_white_noise_filtered() -> None:
    """Shaped noise = the linear convolution of the white stream with the taps (whatever the segment length)."""
    hop, taps = 2000, np.zeros(513, dtype=np.complex64)
    taps[0], taps[200], taps[512] = 1.0, 0.5j, -0.25
    tf = np.asarray(np.fft.fft(taps, 1600), dtype=np.complex64)
    y = _dsp.noise(3, 10, 3, hop, 1000, tf, 512)
    w = _dsp.white(3, 9, 4, hop).astype(np.complex128)  # hops 9 .. 12
    want = np.convolve(w, taps)[hop : 4 * hop]  # output sample n uses white n - 512 .. n
    assert np.max(np.abs(y - want)) < 1e-5


def test_noise_hop_alone_equals_hop_in_a_run() -> None:
    tf = _taps_f(8640)
    run = _dsp.noise(5, B0, 6, 24_000, 8000, tf, 512)
    np.testing.assert_array_equal(_dsp.noise(5, B0 + 5, 1, 24_000, 8000, tf, 512), run[5 * 24_000 :])
    np.testing.assert_array_equal(_dsp.noise(5, B0 + 1, 2, 24_000, 8000, tf, 512), run[24_000 : 3 * 24_000])


def test_adc_is_identical() -> None:
    """Same float input, same bytes: rint half to even (127.5 -> 128), clip at 0 and 255, optional sum first."""
    g = np.random.default_rng(10)
    vals = np.concatenate([g.standard_normal(200_000) * 80, np.arange(-130, 132) - 0.5, [1e30, -1e30, 0.0, -0.0]])
    x = vals.astype(np.float32).view(np.complex64)
    y = (g.standard_normal(len(x) * 2) * 5).astype(np.float32).view(np.complex64)
    for gain in (1.0, 0.7, 26.7):
        np.testing.assert_array_equal(_dsp.adc(x, gain), dsp_ref.adc(x, gain))
        np.testing.assert_array_equal(_dsp.adc(x, gain, y), dsp_ref.adc(x, gain, y))
    u = _dsp.adc(np.array([0, 0.5 + 1.5j, -0.5 - 1.5j, 127.5 + 200j], dtype=np.complex64), 1.0)
    assert u.tolist() == [128, 128, 128, 129, 127, 126, 255, 255]


@pytest.mark.parametrize("impl", [_dsp, dsp_ref], ids=["c++", "numpy"])
def test_out_is_checked(impl) -> None:
    x = np.zeros(10, dtype=np.complex64)
    with pytest.raises(ValueError, match="dtype"):
        impl.adc(x, 1.0, None, np.zeros(20, dtype=np.int16))
    with pytest.raises(ValueError, match="elements"):
        impl.adc(x, 1.0, None, np.zeros(19, dtype=np.uint8))
    with pytest.raises(ValueError, match="contiguous"):
        impl.adc(x, 1.0, None, np.zeros(40, dtype=np.uint8)[::2])
    out = np.zeros(20, dtype=np.uint8)
    assert np.shares_memory(impl.adc(x, 1.0, None, out), out) and np.all(out == 128)


@pytest.mark.parametrize("impl", [_dsp, dsp_ref], ids=["c++", "numpy"])
def test_kernels_reject_bad_arguments(impl) -> None:
    s = [np.zeros(330, dtype=np.complex64)]
    a = [np.ones(2, dtype=np.uint8)]
    with pytest.raises(ValueError, match="same length"):
        impl.synth(s, [220, 220], [0], a, 0, 2, M)
    with pytest.raises(ValueError, match="even"):
        impl.synth(s, [221], [0], a, 0, 2, M)
    with pytest.raises(ValueError, match="samples"):
        impl.synth([np.zeros(100, dtype=np.complex64)], [220], [0], a, 0, 2, M)
    with pytest.raises(ValueError, match="h_rx"):
        impl.synth(s, [220], [0], a, 0, 2, M, np.ones(10, dtype=np.float32))
    tf = np.ones(1600, dtype=np.complex64)
    with pytest.raises(ValueError, match="h0"):
        impl.noise(1, 0, 1, 2000, 1000, tf, 512)
    with pytest.raises(ValueError, match="sub"):
        impl.noise(1, 5, 1, 2000, 300, tf, 512)
    with pytest.raises(ValueError, match="taps_f"):
        impl.noise(1, 5, 1, 2000, 2000, tf, 512)
    with pytest.raises(ValueError, match="hop"):
        impl.white(1, 5, 1, 3)


# --- modem and conv kernels ---

KEY = 0x9E3779B97F4A7C15


@pytest.mark.parametrize("block", [0, 7, 2**33 + 5, 2**64 - 1])
@pytest.mark.parametrize("n", [0, 1, 16, 1020])
def test_random_bytes_is_bit_identical(block: int, n: int) -> None:
    got, want = _dsp.random_bytes(KEY, block, n), dsp_ref.random_bytes(KEY, block, n)
    assert got.dtype == want.dtype == np.uint8 and got.shape == want.shape == (n,)
    np.testing.assert_array_equal(got, want)


def test_random_bytes_are_keyed_by_block_and_prefix_stable() -> None:
    a = _dsp.random_bytes(KEY, 41, 4096)
    np.testing.assert_array_equal(_dsp.random_bytes(KEY, 41, 100), a[:100])  # any length is a prefix
    assert not np.array_equal(_dsp.random_bytes(KEY, 42, 4096), a)
    assert not np.array_equal(_dsp.random_bytes(KEY + 1, 41, 4096), a)
    counts = np.bincount(a, minlength=256)
    assert counts.min() > 0 and abs(float(np.mean(a)) - 127.5) < 5


def test_ccsds_randomiser_sequence() -> None:
    """CCSDS 131.0-B: the pseudo-random sequence starts FF 48 0E C0 9A 0D 70 BC and repeats every 255 bits."""
    for impl in (_dsp, dsp_ref):
        pn = impl.ccsds_randomise(np.zeros(1020, dtype=np.uint8))
        assert pn[:8].tobytes().hex() == "ff480ec09a0d70bc"
        bits = np.unpackbits(pn)
        np.testing.assert_array_equal(bits[255 : 255 + 2000], bits[:2000])
    d = np.random.default_rng(4).integers(0, 256, 1020).astype(np.uint8)
    np.testing.assert_array_equal(_dsp.ccsds_randomise(d), dsp_ref.ccsds_randomise(d))
    np.testing.assert_array_equal(_dsp.ccsds_randomise(_dsp.ccsds_randomise(d)), d)  # an involution


def test_conv_encoder_impulse_response_and_state() -> None:
    """A single 1 gives the generators 171 and 133 (octal) interleaved; encoding in two pieces with the 6-bit
    state carried over equals encoding in one piece."""
    for impl in (_dsp, dsp_ref):
        out = impl.conv_encode(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8), 0)
        g1, g2 = out[0::2][:7], out[1::2][:7]
        assert int("".join(map(str, g1)), 2) == 0o171 and int("".join(map(str, g2)), 2) == 0o133
    bits = np.random.default_rng(5).integers(0, 2, 5000).astype(np.uint8)
    for st in (0, 1, 42, 63):
        np.testing.assert_array_equal(_dsp.conv_encode(bits, st), dsp_ref.conv_encode(bits, st))
    whole = _dsp.conv_encode(bits, 0)
    cut = 1234
    state = int("".join(map(str, bits[cut - 6 : cut])), 2)  # the last 6 bits, newest in bit 0
    np.testing.assert_array_equal(
        np.concatenate([_dsp.conv_encode(bits[:cut], 0), _dsp.conv_encode(bits[cut:], state)]), whole
    )


def _shape_case(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.random.default_rng(seed)
    si = g.choice([-0.7, 0.7], 400).astype(np.float32)
    sq = g.choice([-0.7, 0.7], 400).astype(np.float32)
    return si, sq, g.standard_normal(41).astype(np.float32)


@pytest.mark.parametrize(("sps", "qd"), [(4, 2), (4, 0), (8, 0), (5, 2)])
def test_shape_matches_reference_and_definition(sps: int, qd: int) -> None:
    si, sq, h = _shape_case(sps + qd)
    k0, n0, n = 10**12 + 3, (10**12 + 3) * sps - 30, 400 * sps + 60  # covers both ends of the symbol run
    got = _dsp.shape(si, sq, k0, h, sps, qd, n0, n)
    want = dsp_ref.shape(si, sq, k0, h, sps, qd, n0, n)
    assert _peak_err(got, want) <= RTOL
    c = (len(h) - 1) // 2
    for j in (0, 25, 31, 777, n - 1):  # the definition, sample by sample
        m = n0 + j
        re = sum(
            float(si[k - k0]) * float(h[m - k * sps + c]) for k in range(k0, k0 + 400) if 0 <= m - k * sps + c < 41
        )
        mq = m - qd
        im = sum(
            float(sq[k - k0]) * float(h[mq - k * sps + c]) for k in range(k0, k0 + 400) if 0 <= mq - k * sps + c < 41
        )
        assert abs(got[j] - complex(re, im)) <= 1e-5 * max(1.0, abs(complex(re, im)))


def test_shape_window_is_a_slice() -> None:
    si, sq, h = _shape_case(9)
    k0 = 5 * 10**11
    full = _dsp.shape(si, sq, k0, h, 4, 2, 4 * k0, 1600)
    np.testing.assert_array_equal(_dsp.shape(si, sq, k0, h, 4, 2, 4 * k0 + 333, 500), full[333:833])


def _chirp_case(sf: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    t = 2**sf / 125e3
    starts = np.array([0, 1, 2, 3, 3.25, 4.25]) * t
    lens = np.array([1, 1, 1, 0.25, 1, 1]) * t
    vals = np.array([0, 8, -1, -1, 2**sf - 1, 37], dtype=np.int32)
    return starts, lens, vals, t


@pytest.mark.parametrize("sf", [7, 9, 12])
def test_chirp_matches_reference(sf: int) -> None:
    st, ln, v, t = _chirp_case(sf)
    rate = 250_000.0
    n = int(6 * t * rate) + 50
    got = _dsp.chirp(st, ln, v, sf, 125e3, -1e-4, 1 / rate, n)
    want = dsp_ref.chirp(st, ln, v, sf, 125e3, -1e-4, 1 / rate, n)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-5)
    on = np.abs(got) > 0
    assert np.allclose(np.abs(got[on]), 1, atol=1e-6)
    lead = round(1e-4 * rate)
    assert not on[:lead].any() and on[lead + 1 : lead + int(5.25 * t * rate) - 1].all()  # (sample lead: t ~ 0)


def test_chirp_slope_wrap_and_phase_continuity() -> None:
    """Up-chirps rise at bw / T from -bw/2 + v bw / 2^sf and wrap at +bw/2; down-chirps fall; no phase steps."""
    sf, bw, rate = 8, 125e3, 1e6
    st, ln, v, t = _chirp_case(sf)
    x = _dsp.chirp(st, ln, v, sf, bw, 0.0, 1 / rate, int(6.25 * t * rate)).astype(np.complex128)
    f = np.angle(x[1:] * np.conj(x[:-1])) * rate / (2 * np.pi)  # Hz, between samples j and j + 1
    tm = (np.arange(len(f)) + 0.5) / rate
    slope = bw / t
    for s, (a, d) in enumerate(zip(st, ln, strict=True)):
        sel = (tm > a + 2 / rate) & (tm < a + d - 2 / rate)
        u = tm[sel] - a
        up = (v[s] * bw / 2**sf + slope * u) % bw - bw / 2  # rises from -bw/2 + v bw / 2^sf, wraps at +bw/2
        expect = up if v[s] >= 0 else bw / 2 - slope * u
        err = np.abs((f[sel] - expect + bw / 2) % bw - bw / 2)
        assert np.percentile(err, 99) < 200, (s, float(np.max(err)))
    assert np.max(np.abs(f)) <= bw / 2 + 1e3  # phase continuous at segment edges and wraps


def test_nco_matches_reference() -> None:
    g = np.random.default_rng(6)
    n = 50_001
    gain = g.uniform(0, 3, n)
    cyc = np.cumsum(g.uniform(-0.4, 0.4, n)) + 1e5  # large phase, reduced to [0, 1) cycles inside
    x = _random_iq(n, seed=7)
    for xx in (x, None):
        got, want = _dsp.nco(xx, gain, cyc), dsp_ref.nco(xx, gain, cyc)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-5 * float(np.max(np.abs(want))))
    ref = x.astype(np.complex128) * gain * np.exp(2j * np.pi * cyc)
    np.testing.assert_allclose(_dsp.nco(x, gain, cyc), ref, rtol=0, atol=1e-5 * float(np.max(np.abs(ref))))
    out = np.empty(n, dtype=np.complex64)
    assert np.shares_memory(_dsp.nco(None, gain, cyc, out), out)


@pytest.mark.parametrize("impl", [_dsp, dsp_ref], ids=["c++", "numpy"])
def test_modem_kernels_reject_bad_arguments(impl) -> None:
    f = np.zeros(10, dtype=np.float32)
    with pytest.raises(ValueError, match="n must be"):
        impl.random_bytes(1, 2, -1)
    with pytest.raises(ValueError, match="6 bits"):
        impl.conv_encode(np.zeros(4, dtype=np.uint8), 64)
    with pytest.raises(ValueError, match="same length"):
        impl.shape(f, f[:5], 0, np.ones(5, dtype=np.float32), 4, 2, 0, 10)
    with pytest.raises(ValueError, match="odd"):
        impl.shape(f, f, 0, np.ones(4, dtype=np.float32), 4, 2, 0, 10)
    with pytest.raises(ValueError, match="qd"):
        impl.shape(f, f, 0, np.ones(5, dtype=np.float32), 4, 4, 0, 10)
    st, ln, v, _ = _chirp_case(7)
    with pytest.raises(ValueError, match="sf"):
        impl.chirp(st, ln, v, 13, 125e3, 0.0, 1e-6, 10)
    with pytest.raises(ValueError, match="increase"):
        impl.chirp(st[::-1].copy(), ln, v, 7, 125e3, 0.0, 1e-6, 10)
    with pytest.raises(ValueError, match="2\\^sf"):
        impl.chirp(st, ln, v * 0 + 128, 7, 125e3, 0.0, 1e-6, 10)
    with pytest.raises(ValueError, match="same length"):
        impl.nco(None, np.ones(3), np.ones(4))
