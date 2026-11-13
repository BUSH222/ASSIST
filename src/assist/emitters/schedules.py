"""Schedules and amplitude modulation (plan §6.4).

All schedules are functions of absolute Unix time, so any window can be evaluated on its own:
- `continuous`: always on.
- `on_off`: on for exactly on_s out of every on_s + off_s, starting at phase_s (mod the period). The on interval is
  measured between half-amplitude points; raised-cosine ramps of ramp_ms are centred on its edges.
- `exp_gaps`: bursts separated by min_gap_s + Exp(mean_gap_s - min_gap_s) from the end of one to the start of the
  next. Generated per UTC hour slot from the emitter's keyed stream (source_id, emitter_id, "slot", hour_index);
  each slot starts a fresh chain at slot start + one gap, so slots are independent (random access).
- `poisson_events`: per hour slot, N ~ Poisson(rate for that UTC hour), starts uniform in the hour, durations from
  the configured distribution (clipped to max_s <= 1 h). Same keyed stream per slot. With `min_gap_s` (M7, one
  transmitter) an event starting less than min_gap_s after the previous kept event's end is dropped afterwards (a
  dead time: the kept rate is rate / (1 + rate * (duration + min_gap_s))).

Event signals (LoRa, M6) set an event maker: each event's duration and parameters (spreading factor, payload
length, the frequency index of a multi-frequency emitter) come from it, drawn from the same slot stream in place of
the duration distribution. M7: so do emitters with per-event draws (frequency offset and level), and `ramp_ms`
gives continuous signals under an event schedule raised-cosine ramps centred on each event's edges.
"""

import bisect
import math
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, override

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from assist.rng import RngScope


SLOT_S = 3600
SLOT_CACHE = 6  # hour slots kept per schedule (a VDL station has ~9000 events per hour)


@dataclass(frozen=True)
class Event:
    start: float
    end: float
    params: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class EventMaker:
    """Draws one event's (duration in s, parameters) from the slot stream; upper_s bounds the durations."""

    draw: Callable[[np.random.Generator], tuple[float, dict[str, Any]]]
    upper_s: float


@dataclass(frozen=True)
class DurationDist:
    """Event durations: `fixed` (s), `lognormal` (median_s, sigma) or `exponential` (mean_s), clipped to max_s."""

    dist: str = "fixed"
    s: float = 1.0
    median_s: float = 1.0
    sigma: float = 0.5
    mean_s: float = 1.0
    max_s: float = 3600.0

    @property
    def upper_s(self) -> float:
        return min(self.s, self.max_s) if self.dist == "fixed" else self.max_s

    def draw(self, g: np.random.Generator, n: int) -> np.ndarray:
        if self.dist == "fixed":
            d = np.full(n, self.s)
        elif self.dist == "lognormal":
            d = np.exp(g.normal(math.log(self.median_s), self.sigma, n))
        elif self.dist == "exponential":
            d = g.exponential(self.mean_s, n)
        else:
            raise ValueError(f"unknown duration distribution {self.dist!r}")
        return np.minimum(d, self.max_s)


class Schedule(ABC):
    is_event: bool = False

    @abstractmethod
    def active_fraction(self, t: np.ndarray, step_s: float) -> np.ndarray:
        """Fraction of [t, t + step) during which the emitter is on."""

    @abstractmethod
    def envelope(self, t: np.ndarray) -> np.ndarray:
        """Amplitude envelope (0..1) at t, including ramps."""

    def events(self, t0: float, t1: float) -> list[Event]:
        """Events overlapping [t0, t1) (event schedules only)."""
        return []

    @abstractmethod
    def active_intervals(self, t0: float, t1: float) -> list[tuple[float, float]]:
        """Intervals overlapping [t0, t1) outside which the envelope is exactly 0 (ramps included)."""


class Continuous(Schedule):
    @override
    def active_fraction(self, t: np.ndarray, step_s: float) -> np.ndarray:
        return np.ones(np.shape(t))

    @override
    def envelope(self, t: np.ndarray) -> np.ndarray:
        return np.ones(np.shape(t))

    @override
    def active_intervals(self, t0: float, t1: float) -> list[tuple[float, float]]:
        return [(t0, t1)]


@dataclass
class OnOff(Schedule):
    on_s: float = 40.0
    off_s: float = 180.0
    phase_s: float = 0.0
    ramp_ms: float = 5.0

    @property
    def period_s(self) -> float:
        return self.on_s + self.off_s

    def on_intervals(self, t0: float, t1: float) -> list[tuple[float, float]]:
        """Half-amplitude on intervals overlapping [t0, t1)."""
        p = self.period_s
        k0 = math.floor((t0 - self.phase_s - self.on_s) / p)
        k1 = math.floor((t1 - self.phase_s) / p)
        out = []
        for k in range(k0, k1 + 1):
            a = self.phase_s + k * p
            if a + self.on_s > t0 and a < t1:
                out.append((a, a + self.on_s))
        return out

    def _on_time_until(self, x: np.ndarray) -> np.ndarray:
        """On time accumulated from phase_s up to x (any sign)."""
        p = self.period_s
        k = np.floor((x - self.phase_s) / p)
        u = (x - self.phase_s) - k * p
        return k * self.on_s + np.minimum(u, self.on_s)

    @override
    def active_fraction(self, t: np.ndarray, step_s: float) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        return (self._on_time_until(t + step_s) - self._on_time_until(t)) / step_s

    @override
    def envelope(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        p, r = self.period_s, self.ramp_ms / 1000.0
        u = np.mod(t - self.phase_s + r / 2, p)  # 0 = start of the rising ramp
        env = np.zeros(t.shape)
        if r > 0:
            rise = u < r
            env[rise] = 0.5 * (1 - np.cos(np.pi * u[rise] / r))
            fall = (u >= self.on_s) & (u < self.on_s + r)
            env[fall] = 0.5 * (1 + np.cos(np.pi * (u[fall] - self.on_s) / r))
        env[(u >= r) & (u < self.on_s)] = 1.0
        return env

    @override
    def active_intervals(self, t0: float, t1: float) -> list[tuple[float, float]]:
        h = self.ramp_ms / 2000.0
        return [(a - h, b + h) for a, b in self.on_intervals(t0 - h, t1 + h)]


class _EventSchedule(Schedule):
    is_event = True

    def __init__(
        self,
        scope: RngScope,
        duration: DurationDist,
        maker: EventMaker | None = None,
        ramp_ms: float = 0.0,
    ) -> None:
        self.scope, self.duration = scope, duration  # scope: the emitter's (source_id, emitter_id)
        self.maker = maker
        self.ramp_s = ramp_ms / 1000.0
        self._slots: OrderedDict[int, tuple[list[Event], list[float]]] = OrderedDict()

    @property
    def upper_s(self) -> float:
        """Longest possible event."""
        return self.maker.upper_s if self.maker is not None else self.duration.upper_s

    def _event(self, g: np.random.Generator, t: float) -> Event:
        """An event starting at t, its duration and parameters from the maker (or the duration distribution)."""
        if self.maker is not None:
            d, params = self.maker.draw(g)
        else:
            d = float(self.duration.draw(g, 1)[0])
            params = {"duration_s": d}
        return Event(t, t + d, params)

    @abstractmethod
    def _slot_events(self, g: np.random.Generator, slot: int) -> list[Event]: ...

    def _slot(self, h: int) -> tuple[list[Event], list[float]]:
        hit = self._slots.get(h)
        if hit is None:
            ev = self._slot_events(self.scope.stream("slot", h), h)
            hit = (ev, [e.start for e in ev])  # starts are sorted within a slot
            self._slots[h] = hit
            while len(self._slots) > SLOT_CACHE:
                self._slots.popitem(last=False)
        else:
            self._slots.move_to_end(h)
        return hit

    def slot(self, h: int) -> list[Event]:
        return self._slot(h)[0]

    def prefetch(self, h: int) -> None:
        """Draw hour slot h ahead of need (live serving)."""
        self._slot(h)

    def has_slot(self, h: int) -> bool:
        return h in self._slots

    @override
    def events(self, t0: float, t1: float) -> list[Event]:
        """Events overlapping [t0, t1), ordered by start."""
        lo = t0 - self.upper_s
        out = []
        for h in range(math.floor(lo / SLOT_S), math.floor(t1 / SLOT_S) + 1):
            ev, starts = self._slot(h)
            i0, i1 = bisect.bisect_left(starts, lo), bisect.bisect_left(starts, t1)
            out.extend(e for e in ev[i0:i1] if e.end > t0)
        return sorted(out, key=lambda e: e.start)

    @override
    def active_fraction(self, t: np.ndarray, step_s: float) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        out = np.zeros(t.shape)
        if t.size == 0:
            return out
        for e in self.events(float(t.min()), float(t.max()) + step_s):
            out += np.clip(np.minimum(t + step_s, e.end) - np.maximum(t, e.start), 0, None) / step_s
        return out

    @override
    def active_intervals(self, t0: float, t1: float) -> list[tuple[float, float]]:
        h = self.ramp_s / 2
        return [(e.start - h, e.end + h) for e in self.events(t0 - h, t1 + h)]

    @override
    def envelope(self, t: np.ndarray) -> np.ndarray:
        """1 inside an event, else 0, with raised-cosine ramps of ramp_ms centred on the edges (burst ramps of event
        signals belong to the signal)."""
        t = np.asarray(t, dtype=np.float64)
        out = np.zeros(t.shape)
        if t.size:
            r = self.ramp_s
            for e in self.events(float(t.min()) - r, float(t.max()) + 1.0):
                out = np.maximum(out, event_envelope(t, e.start, e.end, r))
        return out


def event_envelope(t: np.ndarray, start: float, end: float, ramp_s: float) -> np.ndarray:
    """Envelope of one event: 1 on [start, end) (half-amplitude edges), raised-cosine ramps of ramp_s centred on the
    edges (shortened to the event's length), 0 outside."""
    t = np.asarray(t, dtype=np.float64)
    if ramp_s <= 0:
        return ((t >= start) & (t < end)).astype(np.float64)
    r = min(ramp_s, end - start)
    u = np.clip((t - (start - r / 2)) / r, 0, 1)
    d = np.clip(((end + r / 2) - t) / r, 0, 1)
    return 0.5 * (1 - np.cos(np.pi * u)) * 0.5 * (1 - np.cos(np.pi * d)) * (u > 0) * (d > 0)


class ExpGaps(_EventSchedule):
    def __init__(
        self,
        scope: RngScope,
        duration: DurationDist,
        mean_gap_s: float = 180.0,
        min_gap_s: float = 20.0,
        maker: EventMaker | None = None,
        ramp_ms: float = 0.0,
    ) -> None:
        if not 0 <= min_gap_s <= mean_gap_s:
            raise ValueError("exp_gaps needs 0 <= min_gap_s <= mean_gap_s")
        super().__init__(scope, duration, maker, ramp_ms)
        if self.upper_s > min_gap_s:
            raise ValueError("exp_gaps: event durations (max_s) must not exceed min_gap_s, or slots could overlap")
        self.mean_gap_s, self.min_gap_s = mean_gap_s, min_gap_s

    def _gap(self, g: np.random.Generator) -> float:
        return self.min_gap_s + float(g.exponential(self.mean_gap_s - self.min_gap_s))

    @override
    def _slot_events(self, g: np.random.Generator, slot: int) -> list[Event]:
        end_slot = (slot + 1) * SLOT_S
        t = slot * SLOT_S + self._gap(g)
        out = []
        while t < end_slot:
            ev = self._event(g, t)
            out.append(ev)
            t = ev.end + self._gap(g)
        return out


class PoissonEvents(_EventSchedule):
    def __init__(
        self,
        scope: RngScope,
        duration: DurationDist,
        rate_per_hour: float | Sequence[float] = 1.0,
        maker: EventMaker | None = None,
        ramp_ms: float = 0.0,
        min_gap_s: float = 0.0,
    ) -> None:
        super().__init__(scope, duration, maker, ramp_ms)
        if self.upper_s > SLOT_S:
            raise ValueError("poisson_events: max_s must be <= 3600 s")
        self.rate_per_hour = rate_per_hour
        self.min_gap_s = min_gap_s

    def rate(self, slot: int) -> float:
        r = self.rate_per_hour
        return float(r) if isinstance(r, int | float) else float(r[slot % 24])  # UTC hour of day

    @override
    def _slot_events(self, g: np.random.Generator, slot: int) -> list[Event]:
        n = int(g.poisson(self.rate(slot)))
        starts = np.sort(g.uniform(0.0, SLOT_S, n)) + slot * SLOT_S
        if self.maker is not None:
            out = [self._event(g, float(s)) for s in starts]
        else:
            durs = self.duration.draw(g, n)
            out = [Event(float(s), float(s + d), {"duration_s": float(d)}) for s, d in zip(starts, durs, strict=True)]
        if self.min_gap_s > 0:  # one transmitter: drop events that would start too soon (all drawn first: keyed)
            kept: list[Event] = []
            for e in out:
                if not kept or e.start >= kept[-1].end + self.min_gap_s:
                    kept.append(e)
            out = kept
        return out


@dataclass(frozen=True)
class AmModel:
    """Tumbling: gain_dB(t) = -(depth_db / 2) (1 - cos(2 pi t / period_s + phase_rad)), t in absolute Unix seconds."""

    period_s: float
    depth_db: float
    phase_rad: float = 0.0

    def gain_db(self, t: np.ndarray) -> np.ndarray:
        frac = np.fmod(np.asarray(t, dtype=np.float64), self.period_s) / self.period_s  # exact reduction first
        return -(self.depth_db / 2) * (1 - np.cos(2 * np.pi * frac + self.phase_rad))
