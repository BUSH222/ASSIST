"""Per-emitter rendering state (plan §4.1, §4.2): narrow rate and coarse bin, control samples, narrow blocks.

**Control samples** are computed per absolute 1 s chunk at whole Unix milliseconds, so every value depends only on
absolute time, never on where a render starts (window independence, T-WIN):
- amplitude: the link budget (sqrt(mW), no AM or schedule);
- phase, in cycles relative to the coarse bin f_bin = fc + k * 50 Hz (plan §4.1):

      phi(t) = 2 pi * integral(f_tx - f_bin) - 2 pi * f_tx(t) * R(t) / c

  The constant-frequency part of the integral is exact: frequencies are integer mHz and times integer ms, so it is
  reduced modulo one cycle with Python integers. The drift part is (drift / 2) (t - t_aos)^2 from the pass key.
  The range term gives Doppler as exact phase, with no integration drift.

Both are linearly interpolated to the narrow rate. The schedule envelope (with its 5 ms ramps), the AM gain and
the horizon mask are evaluated directly at the narrow rate; the `nco` kernel applies level and phase to the
signal's samples (M6).

**Narrow stream** (M5): per chunk c, one vectorised pass gives the narrow samples of all its OLA blocks 100c ..
100c + 99, i.e. narrow indices [100c L/2, (100c + 101) L/2) (L/2 = rate / 100, so the stream starts at Unix
second c), plus a flag per block saying whether the emitter is active in it. Every value depends only on the chunk
and the absolute index, so the same samples come out whichever window is rendered.

**Event signals** (LoRa, M6): the signal renders the events overlapping the chunk (each whole, from its start);
the schedule envelope is not applied (bursts start and end in the signal). An emitter with several frequencies
gets one EmitterRender per frequency, each rendering only the events on its own.

**Ground and local emitters** (M7): no geometry, so the control samples are a constant amplitude (the received level)
and the phase of the constant offset from the bin. Their events (and those of any emitter with per-event draws)
are rendered one at a time: event signals from their start, continuous signals (cw) under the event's envelope
with ramps, each multiplied by its own level (`level_db`) and shifted by its own frequency offset (`offset_hz`,
phase 0 at the event's start sample), then summed; overlapping events (two transmitters on one channel) simply
add, so they beat.

**Narrow rate** (plan §4.2): r >= 1.25 (B + 2 max|f_rx - f_bin|), a multiple of 100 Hz and of the signal's rate
step (the symbol rate, so samples per symbol are whole), with an FFT-friendly block length. The excursion bound
covers the bin rounding, Doppler at the orbit's perigee speed (plus Earth rotation), 6 sigma of the per-pass
offset and the drift over 30 min. Each chunk checks the actual excursion and fails loudly if it is exceeded.
"""

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.fft as sfft

from assist.emitters.link import C_M_S
from assist.emitters.schedules import event_envelope
from assist.orbits.geometry import BLOCK_S
from assist.signals.base import lcm

if TYPE_CHECKING:
    from types import ModuleType

    from assist.emitters.emitter import Emitter
    from assist.emitters.schedules import Event
    from assist.orbits.geometry import Pass
    from assist.signals.base import Signal

CONTROL_HZ = 1000
HOPS_PER_CHUNK = 100  # one control chunk = 100 hops = 1 s
RATE_MARGIN = 1.25
OFFSET_SIGMAS = 6.0
DRIFT_BOUND_S = 1800.0
PASS_PAD_S = 0.002  # control samples this close outside a pass use that pass's carrier terms
EVENT_PAD_S = 0.01  # per-event rendering: events this close outside a window are checked (filter tails, ramps)


def fade(t: np.ndarray, passes: list[tuple[float, float]], fade_s: float) -> np.ndarray:
    """Visibility as an amplitude gain: 0 outside the passes; inside, a raised-cosine rise from 0 to 1 over the
    `fade_s` seconds after each pass's AOS and the mirror fall over the `fade_s` seconds before its LOS (plan §6.2: a
    fade from and to pure noise instead of a horizon map). A pass shorter than 2 `fade_s` peaks below 1. A pure
    function of t and the pass, so every window gives the same values."""
    g = np.zeros(t.shape)
    for a, z in passes:
        sel = (t >= a) & (t < z)
        if fade_s > 0:
            x = np.minimum(np.minimum(t[sel] - a, z - t[sel]) / fade_s, 1.0)
            g[sel] = 0.5 - 0.5 * np.cos(np.pi * x)
        else:
            g[sel] = 1.0
    return g


class RenderError(RuntimeError):
    pass


def narrow_rate(bandwidth_hz: float, excursion_hz: float, preferred: int, fs: int, m: int, step: int = 100) -> int:
    """Smallest rate (a multiple of lcm(100, step) Hz, >= preferred) meeting the narrow-rate rule, with a fast FFT
    length."""
    q = lcm(100, int(step))
    r = max(int(preferred), math.ceil(RATE_MARGIN * (bandwidth_hz + 2 * excursion_hz)), 100)
    r = -(-r // q) * q
    while True:
        if r > fs:
            raise RenderError(f"narrow rate {r} Hz would exceed the sample rate {fs} Hz")
        n = r * m // fs
        if r * m % fs == 0 and n % 2 == 0 and sfft.next_fast_len(n) == n:
            return r
        r += q


def _intersect(xs: list[tuple[float, float]], ys: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out = []
    for a0, a1 in xs:
        for b0, b1 in ys:
            lo, hi = max(a0, b0), min(a1, b1)
            if lo < hi:
                out.append((lo, hi))
    return out


@dataclass(frozen=True)
class Control:
    k0: int  # first control sample, integer Unix ms
    amp: np.ndarray  # link amplitude, sqrt(mW)
    cyc: np.ndarray  # carrier phase in cycles relative to f_bin (a whole number of cycles removed)
    visible: list[tuple[float, float]]  # (aos, los) of the passes overlapping the chunk
    active: list[tuple[float, float]]  # where the emitter may be non-zero: visible and schedule active


@dataclass(frozen=True)
class Stream:
    """An emitter's narrow samples for the OLA blocks of one chunk (plan §4.2), before the Hann window."""

    x: np.ndarray  # complex64: 101 * L / 2 samples from narrow index 100 c L / 2 (blocks 100c .. 100c + 99)
    active: np.ndarray  # uint8 per block: block 100c + i is synthesised only where active[i]


class EmitterRender:
    """One emitter (one of its frequencies) as the synthesiser sees it: rate, bin, and its narrow samples for any
    absolute chunk."""

    def __init__(
        self,
        em: Emitter,
        signal: Signal,
        fc_hz: float,
        fs: int,
        hop: int,
        m: int,
        dsp: ModuleType,
        channel: int = 0,
    ) -> None:
        self.em, self.signal, self.dsp = em, signal, dsp
        self.fs, self.hop, self.m = fs, hop, m
        self.channel = channel
        self.multi = len(em.channels_mhz) > 1
        self.hop_ms = hop * 1000 // fs
        self.chunk_ms = HOPS_PER_CHUNK * self.hop_ms
        bin_mhz = fs * 1000 // m  # 50 Hz in mHz
        fc_mhz = round(fc_hz * 1000)
        rest = self.rest_mhz = em.channels_mhz[channel]
        self.k_bin = (rest - fc_mhz + bin_mhz // 2) // bin_mhz
        self.f_bin_mhz = fc_mhz + self.k_bin * bin_mhz
        car = em.carrier
        self.fixed = em.fixed
        # Ground/local emitters, and any with per-event draws, render their events one at a time (M7).
        self.per_event = em.per_event or (self.fixed and em.is_event)
        excursion = abs(rest - self.f_bin_mhz) / 1000.0 + em.max_event_offset_hz
        if not self.fixed:
            excursion += (
                rest / 1000.0 * em.platform.max_speed_km_s() * 1000.0 / C_M_S
                + OFFSET_SIGMAS * (car.pass_offset_sigma_hz or 0.0)
                + (car.drift_hz_s or 0.0) * DRIFT_BOUND_S
            )
        self.bandwidth_hz = signal.bandwidth_hz()
        self.rate = narrow_rate(self.bandwidth_hz, excursion, signal.preferred_rate(), fs, m, signal.rate_step())
        self.n = self.rate * m // fs  # narrow samples per block
        self.limit_hz = (self.rate / RATE_MARGIN - self.bandwidth_hz) / 2
        if self.fixed and abs(rest - self.f_bin_mhz) / 1000.0 + em.max_event_offset_hz > self.limit_hz:
            raise RenderError(f"{self.id}: frequency excursion exceeds {self.limit_hz:.1f} Hz")
        self._ctl: OrderedDict[int, Control] = OrderedDict()
        self._streams: OrderedDict[int, Stream | None] = OrderedDict()
        self._hours: OrderedDict[int, list[Pass]] = OrderedDict()

    @property
    def id(self) -> str:
        base = f"{self.em.source_id}/{self.em.id}"
        return f"{base}@{self.rest_mhz / 1e9:.6f}MHz" if self.multi else base

    def events(self, t0: float, t1: float) -> list[Event]:
        """Event signals: this frequency's events overlapping [t0, t1)."""
        return [e for e in self.em.schedule.events(t0, t1) if self.em.event_channel(e) == self.channel]

    def _hour_passes(self, hr: int) -> list[Pass]:
        """Complete passes overlapping UTC hour hr (geometry for that hour, and the next if a pass runs on)."""
        ps = self._hours.get(hr)
        if ps is None:
            ps = self.em.platform.passes(float(hr * BLOCK_S), float((hr + 1) * BLOCK_S))
            self._hours[hr] = ps
            while len(self._hours) > 4:
                self._hours.popitem(last=False)
        return ps

    def has_hour(self, hr: int) -> bool:
        return hr in self._hours

    def prefetch_hour(self, hr: int) -> None:
        """Compute hour hr's passes (and the geometry under them) ahead of need (live serving, plan §4.5)."""
        self._hour_passes(hr)

    def _passes(self, t0: float, t1: float) -> list[Pass]:
        """Complete passes overlapping [t0, t1), computed once per UTC hour."""
        found: dict[float, Pass] = {}
        for hr in range(math.floor(t0 / BLOCK_S), math.floor(t1 / BLOCK_S) + 1):
            for p in self._hour_passes(hr):
                if p.los > t0 and p.aos < t1:
                    found[p.aos] = p
        return [found[k] for k in sorted(found)]

    def control(self, chunk: int) -> Control:
        ctl = self._ctl.get(chunk)
        if ctl is None:
            ctl = self._compute_control(chunk)
            self._ctl[chunk] = ctl
            while len(self._ctl) > 2:
                self._ctl.popitem(last=False)
        return ctl

    def event_span(self, ev: Event) -> tuple[float, float]:
        """Where an event's samples may be non-zero: the signal's bounds (event signals), else the event and ramps."""
        if self.signal.mode == "event":
            return self.signal.event_bounds(ev)
        h = float(getattr(self.em.schedule, "ramp_s", 0.0)) / 2
        return ev.start - h, ev.end + h

    def _on(self, t_lo: float, t_hi: float) -> list[tuple[float, float]]:
        """Intervals where the emitter's schedule (or its events) may be non-zero, overlapping [t_lo, t_hi]."""
        if self.per_event:
            evs = self.events(t_lo - EVENT_PAD_S, t_hi + EVENT_PAD_S)
            return [self.event_span(e) for e in evs]
        if self.signal.mode == "event":
            return [(e.start, e.end) for e in self.events(t_lo, t_hi + 1e-3)]
        return self.em.schedule.active_intervals(t_lo, t_hi + 1e-3)

    def _control_fixed(self, chunk: int) -> Control:
        """Ground/local: constant amplitude, the phase of the constant offset from the bin (exact, integers)."""
        k0 = chunk * self.chunk_ms
        k = np.arange(k0, k0 + self.chunk_ms + 2 * self.hop_ms + 1, dtype=np.int64)
        t_lo, t_hi = k0 / 1000.0, float(k[-1]) / 1000.0
        rx = self.em.rx_dbm
        amp = np.full(k.shape, 0.0 if rx is None else 10 ** (rx / 20))
        delta = self.rest_mhz - self.f_bin_mhz
        r0 = (delta * k0) % 1_000_000
        cyc = (r0 + delta * (k - k0)) / 1e6
        cyc -= math.floor(cyc[0])
        visible = [(t_lo - 1.0, t_hi + 1.0)]
        return Control(int(k0), amp, cyc, visible, _intersect(visible, self._on(t_lo, t_hi)))

    def _compute_control(self, chunk: int) -> Control:
        if self.fixed:
            return self._control_fixed(chunk)
        em, car = self.em, self.em.carrier
        k0 = chunk * self.chunk_ms
        k = np.arange(k0, k0 + self.chunk_ms + 2 * self.hop_ms + 1, dtype=np.int64)  # covers the chunk's blocks
        t = k / 1000.0
        t_lo, t_hi = float(t[0]), float(t[-1])
        passes = self._passes(t_lo - 1.0, t_hi + 1.0)
        key = np.full(k.shape, -1, dtype=np.int64)
        base = np.full(k.shape, self.rest_mhz, dtype=np.int64)
        drift = np.zeros(k.shape)
        for p in passes:
            if p.aos_key is None:
                raise RenderError(f"{self.id}: pass without a key")
            sel = (t >= p.aos - PASS_PAD_S) & (t < p.los + PASS_PAD_S)
            off, d = car.pass_terms(p.aos_key)
            key[sel] = p.aos_key
            base[sel] = self.rest_mhz + off
            drift[sel] = d
        g = em.platform.geometry_ms(k)
        amp = 10 ** (em.link_dbm(g.el_deg, g.az_deg, g.range_m, g.off_nadir_deg, self.rest_mhz) / 20)
        # Constant-frequency part: (f - f_bin)[mHz] * t[ms] / 1e6 cycles, reduced mod 1 cycle with integers.
        delta = base - self.f_bin_mhz
        i = k - k0
        cyc = np.empty(k.shape)
        for dv in np.unique(delta):
            sel = delta == dv
            r0 = (int(dv) * k0) % 1_000_000
            cyc[sel] = (r0 + int(dv) * i[sel]) / 1e6
        dt_aos = np.where(key >= 0, (k - 1000 * key) / 1000.0, 0.0)
        f_tx = base / 1000.0 + drift * dt_aos
        cyc += 0.5 * drift * dt_aos**2 - f_tx * g.range_m / C_M_S
        cyc -= math.floor(cyc[0])
        visible = [(p.aos, p.los) for p in passes]
        active = _intersect(visible, self._on(t_lo, t_hi))
        self._check_excursion(t, cyc, active)
        return Control(int(k0), amp, cyc, visible, active)

    def _check_excursion(self, t: np.ndarray, cyc: np.ndarray, active: list[tuple[float, float]]) -> None:
        if not active:
            return
        f = np.diff(cyc) * CONTROL_HZ  # Hz relative to f_bin, at the midpoints
        tm = (t[:-1] + t[1:]) / 2
        on = np.zeros(tm.shape, dtype=bool)
        for a, b in active:
            on |= (tm >= a) & (tm < b)
        if on.any():
            worst = float(np.max(np.abs(f[on]))) + self.em.max_event_offset_hz
            if worst > self.limit_hz:
                raise RenderError(
                    f"{self.id}: frequency excursion {worst:.1f} Hz from the bin exceeds {self.limit_hz:.1f} Hz "
                    f"(narrow rate {self.rate} Hz)"
                )

    def stream(self, chunk: int) -> Stream | None:
        """Narrow samples of chunk `chunk`'s blocks, or None when the emitter is silent in all of them."""
        if chunk in self._streams:
            return self._streams[chunk]
        st = self._compute_stream(chunk)
        self._streams[chunk] = st
        while len(self._streams) > 2:
            self._streams.popitem(last=False)
        return st

    def _compute_stream(self, chunk: int) -> Stream | None:
        ctl = self.control(chunk)
        b = HOPS_PER_CHUNK * chunk + np.arange(HOPS_PER_CHUNK, dtype=np.int64)
        tb0 = b * self.hop / self.fs
        tb1 = (b * self.hop + self.m) / self.fs
        active = np.zeros(HOPS_PER_CHUNK, dtype=bool)
        for a, z in ctl.active:
            active |= (tb0 < z) & (tb1 > a)
        if not active.any():
            return None
        half = self.n // 2
        k = (HOPS_PER_CHUNK + 1) * half
        n0 = HOPS_PER_CHUNK * chunk * half  # narrow index of Unix second `chunk`
        j = np.arange(k, dtype=np.int64)
        t = (n0 + j) / self.rate
        pos = j * (1000.0 / self.rate)  # ms from the chunk's first control sample (k0 = 1000 chunk)
        grid = np.arange(len(ctl.amp), dtype=np.float64)
        amp = np.interp(pos, grid, ctl.amp)
        cyc = np.interp(pos, grid, ctl.cyc)
        vis = fade(t, ctl.visible, self.em.fade_s)
        if self.per_event:
            env = vis
            x = self._events_x(n0, k, float(t[0]), float(t[-1]))
        elif self.signal.mode == "event":
            env = vis
            x = self.signal.render_events(self.events(float(t[0]), float(t[-1]) + 1e-3), n0, k, self.rate)
        else:
            env = self.em.schedule.envelope(t) * vis
            x = None if self.signal.constant else self.signal.render(n0, k, self.rate)
        if self.em.am is not None:
            env = env * 10 ** (self.em.am.gain_db(t) / 20)
        v = self.dsp.nco(x, amp * env, cyc)
        return Stream(v, active.astype(np.uint8))

    def _events_x(self, n0: int, k: int, t_lo: float, t_hi: float) -> np.ndarray:
        """Narrow samples [n0, n0 + k) of the events touching [t_lo, t_hi], one at a time, each with its own level and
        frequency offset (M7)."""
        x = np.zeros(k, dtype=np.complex64)
        rate = self.rate
        ramp = float(getattr(self.em.schedule, "ramp_s", 0.0))
        for ev in self.events(t_lo - EVENT_PAD_S, t_hi + EVENT_PAD_S):
            b0, b1 = self.event_span(ev)
            j0 = max(0, math.ceil(b0 * rate) - n0)
            j1 = min(k, math.floor(b1 * rate) - n0 + 1)
            if j1 <= j0:
                continue
            m = j1 - j0
            if self.signal.mode == "event":
                xe = self.signal.render_events([ev], n0 + j0, m, rate)
            else:
                idx = n0 + np.arange(j0, j1, dtype=np.int64)
                env = event_envelope(idx / rate, ev.start, ev.end, ramp)
                xe = env.astype(np.complex64) if self.signal.constant else self.signal.render(n0 + j0, m, rate) * env
            g = 10 ** (float(ev.params.get("level_db", 0.0)) / 20)
            off = float(ev.params.get("offset_hz", 0.0))
            if off:
                dn = n0 + np.arange(j0, j1, dtype=np.int64) - round(ev.start * rate)
                xe = xe * (g * np.exp(2j * np.pi * (off * dn / rate)))
            elif g != 1.0:
                xe = xe * g
            x[j0:j1] += xe.astype(np.complex64)
        return x
