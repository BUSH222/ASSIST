"""T-SIG and T-SCH (plan §5, §10.1): the M6 satellite signals on their own, at their narrow rate, noiseless.

- oqpsk (METEOR LRPT): unit power; occupied bandwidth rs (1 + alpha) and fitted alpha; the 4th-power line used by
  ADRIFT's suppressed-carrier branch; CADU framing (ASM, randomiser, convolutional code across frames); frames never
  repeat.
- sdpsk (ORBCOMM subscriber): every transition +-90 deg after the matched filter; the M1 estimators (P2b) find
  alpha 0.4 and 4800 sym/s.
- oqpsk_burst (ORBCOMM gateway): block edges within 0.1 ms of the schedule, off time 627-692 ms, 4 or 5 blocks.
- lora: Semtech airtime, burst length, chirp slope and structure; schedule draws (sf, payload, channel) keyed.
Every kind: any window is the same slice of a longer one (T-WIN).
"""

import itertools
import math
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.optimize import least_squares
from scipy.signal import resample_poly, welch

from assist import _dsp, dsp_ref
from assist.emitters.schedules import DurationDist, Event, EventMaker, ExpGaps
from assist.profile import orbcomm as p2b
from assist.profile.dsp import rc_psd
from assist.rng import KeyedRng
from assist.signals import lora as L
from assist.signals.base import Keys, check_params, event_maker, make_signal
from assist.signals.oqpsk import ASM, FRAME_SYMBOLS
from assist.signals.pulse import rrc_taps

T0 = 1_793_793_600  # 2026-11-04T12:00:00Z
KEYS = Keys(KeyedRng(11), "src", "em")


def _sig(kind: str, params: dict[str, Any] | None = None, dsp=_dsp):
    return make_signal(kind, params or {}, KEYS, dsp)


def _psd(x: np.ndarray, rate: float, nper: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    f, p = welch(x, fs=rate, nperseg=nper, return_onesided=False, scaling="density")
    o = np.argsort(f)
    return f[o], p[o]


def _fit_alpha(x: np.ndarray, rate: float, rs: float) -> float:
    """alpha of A * RC(f; rs, alpha) fitted to the PSD in dB (noiseless, so the fit runs down to -40 dB)."""
    f, p = _psd(x, rate)
    sel = (np.abs(f) < rs) & (p > p.max() * 1e-4)

    def resid(v: np.ndarray) -> np.ndarray:
        return 10 * np.log10(v[0] * rc_psd(f[sel], rs, v[1]) + 1e-30) - 10 * np.log10(p[sel])

    return float(least_squares(resid, [float(p.max()), 0.5], bounds=([0, 0.05], [np.inf, 1.0])).x[1])


# --- parameters ---


def test_params_get_defaults_and_reject_typos() -> None:
    assert check_params("oqpsk", {}) == {"rs": 72000, "alpha": 0.6, "sps": 4, "span": 10, "framing": "meteor_cadu"}
    assert check_params("sdpsk", {})["rs"] == 4800 and check_params("sdpsk", {})["alpha"] == 0.4
    assert check_params("oqpsk_burst", {})["blocks"] == [4, 5]
    assert check_params("lora", {})["sf"] == [7, 12] and check_params("lora", {})["payload_bytes"] == [10, 60]
    with pytest.raises(ValidationError, match="extra"):
        check_params("oqpsk", {"rolloff": 0.5})
    with pytest.raises(ValidationError, match="span must be even"):
        check_params("sdpsk", {"span": 9})
    with pytest.raises(ValidationError, match="sf must be"):
        check_params("lora", {"sf": [7, 13]})
    with pytest.raises(ValidationError, match="whole number of symbols"):
        check_params("oqpsk_burst", {"block_ms": 62.51})
    with pytest.raises(ValidationError, match="do not fit"):
        check_params("oqpsk_burst", {"blocks": [4, 16]})


def test_rrc_taps() -> None:
    h = rrc_taps(0.6, 4, 10)
    assert len(h) == 41 and np.isclose(np.sum(h**2), 4) and np.allclose(h, h[::-1])
    rc = np.convolve(h, h)[::4]  # RRC * RRC is Nyquist: zero at every other symbol instant
    centre = len(rc) // 2
    assert np.max(np.abs(np.delete(rc, centre))) < 0.01 * rc[centre]


# --- window independence and backends ---


@pytest.mark.parametrize("kind", ["oqpsk", "sdpsk", "oqpsk_burst"])
def test_window_is_a_slice_and_backends_agree(kind: str) -> None:
    s = _sig(kind)
    rate = s.preferred_rate()
    n0 = T0 * rate + 12_345
    full = s.render(n0, rate, rate)
    a, m = rate // 3 + 7, rate // 5
    np.testing.assert_array_equal(s.render(n0 + a, m, rate), full[a : a + m])
    ref = _sig(kind, dsp=dsp_ref).render(n0, rate, rate)
    assert np.max(np.abs(ref - full)) <= 1e-5 * np.max(np.abs(full))


def test_rate_must_be_a_multiple_of_the_symbol_rate() -> None:
    s = _sig("oqpsk")
    assert s.rate_step() == 144_000  # even samples per symbol (Q delay)
    with pytest.raises(ValueError, match="multiple of rs"):
        s.render(0, 10, 360_000)  # 5 samples per symbol


# --- oqpsk (METEOR LRPT) ---


@pytest.fixture(scope="module")
def meteor() -> tuple[np.ndarray, int]:
    s = _sig("oqpsk")
    rate = s.preferred_rate()
    return s.render(T0 * rate, 2 * rate, rate), rate


def test_oqpsk_power_bandwidth_and_alpha(meteor: tuple[np.ndarray, int]) -> None:
    x, rate = meteor
    assert abs(float(np.mean(np.abs(x) ** 2)) - 1) < 0.01
    f, p = _psd(x, rate)
    edge = 72_000 * 1.6 / 2
    outside = float(np.sum(p[np.abs(f) > edge + 2000]) / np.sum(p))
    assert outside < 1e-3  # occupied bandwidth rs (1 + alpha) = 115.2 kHz
    assert abs(_fit_alpha(x, rate, 72_000) - 0.6) <= 0.05


def test_oqpsk_fourth_power_line(meteor: tuple[np.ndarray, int]) -> None:
    """x^4 has a line at 4 f_c (here 0 Hz) standing well above the rest of its spectrum."""
    x, rate = meteor
    z = x.astype(np.complex128) ** 4
    s = np.abs(np.fft.fft(z * np.hanning(len(z)))) ** 2
    f = np.fft.fftfreq(len(z), 1 / rate)
    peak = int(np.argmax(s))
    assert abs(f[peak]) < 1.0
    assert 10 * np.log10(s[peak] / np.median(s)) > 40


def test_meteor_cadu_framing() -> None:
    s = _sig("oqpsk")
    f = (T0 * 72_000) // FRAME_SYMBOLS
    cadu = s.cadu(f)
    assert len(cadu) == 1024 and cadu[:4].tobytes() == ASM.tobytes()
    # The payload is random bytes through the CCSDS randomiser; frames never repeat.
    assert not np.array_equal(s.payload(f), s.payload(f + 1))
    raw = _dsp.random_bytes(KEYS.key64("frame"), f, 1020)
    np.testing.assert_array_equal(_dsp.ccsds_randomise(s.payload(f)), raw)
    # The convolutional code runs on across frames: two frames coded in one go = each coded from the previous end.
    bits = np.concatenate([np.unpackbits(s.cadu(f)), np.unpackbits(s.cadu(f + 1))])
    state = int(s.payload(f - 1)[-1]) & 0x3F
    both = _dsp.conv_encode(bits, state)
    np.testing.assert_array_equal(both[: 2 * 8192], s.coded(f))
    np.testing.assert_array_equal(both[2 * 8192 :], s.coded(f + 1))
    # Symbols: coded bit pairs as +-1/sqrt(2) on I and Q; a window across the frame boundary is seamless.
    k = (f + 1) * FRAME_SYMBOLS
    si, sq = s.symbols(k - 3, k + 3)
    c0, c1 = s.coded(f), s.coded(f + 1)
    want_i = np.concatenate([c0[0::2][-3:], c1[0::2][:3]])
    np.testing.assert_allclose(si, (1 - 2 * want_i.astype(np.float32)) / np.sqrt(2), rtol=1e-6)
    assert np.all(np.abs(sq) == np.float32(1 / np.sqrt(2)))


def test_oqpsk_random_framing_differs_and_is_keyed() -> None:
    a = make_signal("oqpsk", {"framing": "random"}, KEYS, _dsp)
    b = make_signal("oqpsk", {"framing": "random"}, Keys(KeyedRng(11), "src", "other"), _dsp)
    n0 = T0 * 288_000
    xa, xb = a.render(n0, 5000, 288_000), b.render(n0, 5000, 288_000)
    assert abs(float(np.mean(np.abs(xa) ** 2)) - 1) < 0.05 and not np.allclose(xa, xb)


# --- sdpsk (ORBCOMM subscriber) ---


def test_sdpsk_transitions_are_plus_minus_90_degrees() -> None:
    s = _sig("sdpsk")
    rate, sps = 38_400, 8
    k0 = T0 * 4800
    x = s.render(k0 * sps, 4000 * sps, rate).astype(np.complex128)
    h = rrc_taps(0.4, sps, 10)
    y = np.convolve(x, h)[len(h) // 2 :][: len(x)] / sps  # matched filter, aligned to the input
    sym = y[0::sps][20:-20]  # symbol centres (symbol k at sample k sps), away from the edges
    d = np.degrees(np.angle(sym[1:] * np.conj(sym[:-1])))
    assert np.all(np.abs(np.abs(d) - 90) < 15)
    states = s.states(k0 + 20, k0 + 20 + len(sym))
    want = np.where(np.diff(states) % 4 == 1, 90.0, -90.0)
    np.testing.assert_array_equal(np.sign(d), np.sign(want))  # the steps carry the data
    assert abs(float(np.mean(np.abs(x) ** 2)) - 1) < 0.02


def test_sdpsk_matches_the_p2b_reference_estimators() -> None:
    """The M1 estimators (symbol rate from |x|^2, RC fit to the PSD) at 50 kS/s with noise: alpha 0.4 +- 0.02,
    4800 sym/s within 0.01 %."""
    s = _sig("sdpsk")
    x = s.render(T0 * 38_400, 60 * 38_400, 38_400)
    y = resample_poly(x.astype(np.complex128), 125, 96)  # 38.4 -> 50 kS/s
    g = np.random.default_rng(1)
    noise = (g.standard_normal(len(y)) + 1j * g.standard_normal(len(y))) * np.sqrt(0.5 * 0.05)  # ~ +10 dB peak
    xc = (y + noise).astype(np.complex64)
    rs = p2b.symbol_rate(xc)
    assert abs(rs - 4800) <= 0.48
    pulse = p2b.fit_pulse(xc, rs)
    assert abs(pulse["alpha"] - 0.4) <= 0.02


# --- oqpsk_burst (ORBCOMM gateway) ---


def test_gateway_burst_timing() -> None:
    s = _sig("oqpsk_burst")
    rate = s.preferred_rate()
    secs = 20
    x = s.render(T0 * rate, secs * rate, rate)
    env = np.abs(x)
    # Half-amplitude crossings of the envelope smoothed over 0.2 ms (|x| itself has OQPSK ripple)
    k = 23
    sm = np.convolve(env, np.ones(k) / k, mode="same")
    level = float(np.median(sm[sm > 0.5]))
    i = np.flatnonzero(np.diff((sm > 0.5 * level).astype(np.int8)))
    t_edges = (i + (0.5 * level - sm[i]) / (sm[i + 1] - sm[i])) / rate + T0
    want = s.on_intervals(T0, T0 + secs)
    want = [(a, b) for a, b in want if a > T0 + 0.01 and b < T0 + secs - 0.01]
    w_edges = np.sort(np.concatenate([[a for a, _ in want], [b for _, b in want]]))
    assert len(t_edges) == len(w_edges)
    assert np.max(np.abs(t_edges - w_edges)) < 0.02e-3  # +-0.1 ms required
    # Blocks of 62.5 ms, 2.5 ms apart, 4 or 5 per 0.95 s segment; off 627.5 or 692.5 ms between segments.
    starts = np.array([a for a, _ in want])
    gaps = np.diff(starts)
    assert np.allclose(np.sort(np.unique(np.round(gaps, 4)))[:1], [0.065])
    seg_first = starts[np.concatenate([[True], gaps > 0.1])]
    n_blocks = np.array([np.sum((starts >= a) & (starts < a + 0.9)) for a in seg_first])
    assert set(n_blocks[1:-1]) == {4, 5}
    ends = np.array([b for _, b in want])
    off = np.array([seg_first[i + 1] - ends[ends < seg_first[i + 1]].max() for i in range(len(seg_first) - 1)])
    assert np.all((off > 0.627) & (off < 0.693))
    assert np.all(np.round(seg_first * 28_800).astype(np.int64) % 27_360 == 0)  # segment s at s * 0.95 s (Unix)
    # Unit power while on
    inside = np.zeros(len(x), dtype=bool)
    for a, b in want:
        inside[int((a - T0) * rate) + 200 : int((b - T0) * rate) - 200] = True
    assert abs(float(np.mean(env[inside] ** 2)) - 1) < 0.05


# --- lora ---


def test_lora_airtime_formula() -> None:
    p = L.LoraParams()
    # Semtech's calculator: SF7, 125 kHz, 10 bytes, CR 4/5, CRC, 8 preamble symbols: 41.216 ms (28 payload symbols).
    assert L.payload_symbols(p, 7, 10) == 28 and math.isclose(L.airtime_s(p, 7, 10), 0.041216)
    # SF12 needs low-data-rate optimisation (T_sym 32.8 ms > 16 ms): 60 bytes -> 68 symbols, 2.629632 s.
    assert L.low_rate(p, 12) and not L.low_rate(p, 10)
    assert L.payload_symbols(p, 12, 60) == 68 and math.isclose(L.airtime_s(p, 12, 60), 2.629632)
    assert L.payload_symbols(p, 9, 51) == 8 + math.ceil((408 - 36 + 44) / 36) * 5
    assert math.isclose(L.LoRa.max_duration_s(p), 2.629632)


@pytest.mark.parametrize("sf", [7, 10, 12])
def test_lora_burst_length_structure_and_slope(sf: int) -> None:
    s = _sig("lora")
    rate = 250_000
    start = T0 + 0.123_456_7
    pl = 23
    d = L.airtime_s(s.p, sf, pl)
    ev = Event(start, start + d, {"sf": sf, "payload_bytes": pl, "duration_s": d})
    n0 = T0 * rate
    x = s.render_events([ev], n0, math.ceil((d + 0.3) * rate), rate)
    on = np.flatnonzero(np.abs(x) > 0)
    assert abs((n0 + on[0]) / rate - start) <= 1 / rate and abs((n0 + on[-1] + 1) / rate - (start + d)) <= 1 / rate
    assert np.allclose(np.abs(x[on]), 1, atol=1e-6)
    # Dechirp each symbol slot against the base up-chirp: the preamble gives bin 0, the sync words 8 and 16,
    # payload values; the down-chirps give no clean peak against an up-chirp.
    n_sym = 1 << sf
    t_sym = n_sym / 125e3
    spp = round(t_sym * rate)  # samples per symbol (an integer here)
    base = _dsp.chirp(np.array([0.0]), np.array([t_sym]), np.array([0], np.int32), sf, 125e3, 0.0, 1 / rate, spp)
    i0 = on[0]
    st, _, val, _ = s.segments(ev)

    def peak(slot_start_s: float) -> int:
        j = i0 + round(slot_start_s * rate)
        z = x[j : j + spp] * np.conj(base)
        spec = np.abs(np.fft.fft(z))
        b = int(np.argmax(spec))  # bin b of an spp-point FFT at `rate` = b * rate / spp Hz
        return round(float(b * rate / spp % 125e3) / (125e3 / n_sym)) % n_sym

    for i in (0, 7, 8, 9, 13, 14, len(st) - 1):
        if val[i] >= 0:
            assert peak(st[i]) == val[i], i
    assert val[10] == val[11] == val[12] == -1
    # Instantaneous frequency during the preamble: slope bw / T_sym from -bw/2
    seg = x[i0 + 5 : i0 + spp - 5].astype(np.complex128)
    fi = np.angle(seg[1:] * np.conj(seg[:-1])) * rate / (2 * np.pi)
    slope = np.polyfit(np.arange(len(fi)) / rate, fi, 1)[0]
    assert abs(slope / (125e3 / t_sym) - 1) < 1e-3


def test_lora_schedule_draws_are_keyed_and_random_access() -> None:
    cls, p = event_maker("lora", {})
    assert isinstance(p, L.LoraParams)
    upper = cls.max_duration_s(p)
    assert upper is not None

    def draw(g: np.random.Generator) -> tuple[float, dict[str, Any]]:
        d, prm = cls.draw_event(p, g)
        assert d is not None
        return d, prm

    maker = EventMaker(draw, upper)
    sch = ExpGaps(KeyedRng(3).scope("s", "lora"), DurationDist(), 180.0, 20.0, maker)
    evs = sch.events(T0, T0 + 6 * 3600)
    assert len(evs) > 60
    for e in evs:
        d = L.airtime_s(p, e.params["sf"], e.params["payload_bytes"])
        assert math.isclose(e.end - e.start, d, abs_tol=1e-6) and e.params["duration_s"] == d
        assert 7 <= e.params["sf"] <= 12 and 10 <= e.params["payload_bytes"] <= 60
    assert len({e.params["sf"] for e in evs}) == 6
    gaps = np.array([b.start - a.end for a, b in itertools.pairwise(evs)])
    assert np.all(gaps[gaps > 0] >= 20 - 1e-9)
    mid = sch.events(T0 + 7200, T0 + 9000)  # a later window on its own gives the same events
    again = ExpGaps(KeyedRng(3).scope("s", "lora"), DurationDist(), 180.0, 20.0, maker).events(T0 + 7200, T0 + 9000)
    assert [(e.start, e.params) for e in mid] == [(e.start, e.params) for e in again]
    with pytest.raises(ValueError, match="min_gap_s"):
        ExpGaps(KeyedRng(3).scope("s", "lora"), DurationDist(), 180.0, 2.0, maker)  # SF12 bursts last up to 2.6 s
