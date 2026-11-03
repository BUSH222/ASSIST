"""Schedules and amplitude modulation (plan §6.4).

All schedules are functions of absolute Unix time, so any window can be evaluated on its own:
- `continuous`: always on.
- `on_off`: on for exactly on_s out of every on_s + off_s, starting at phase_s (mod the period). The on interval is
  measured between half-amplitude points; raised-cosine ramps of ramp_ms are centred on its edges.
- `exp_gaps`: bursts separated by min_gap_s + Exp(mean_gap_s - min_gap_s) from the end of one to the start of the
  next. Generated per UTC hour slot from the keyed stream (source_id, emitter_id, "slot", hour_index); each slot
  starts a fresh chain at slot start + one gap, so slots are independent (random access).
- `poisson_events`: per hour slot, N ~ Poisson(rate for that UTC hour), starts uniform in the hour, durations from
  the configured distribution (clipped to max_s <= 1 h). Same keyed stream per slot.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, override

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from assist.rng import KeyedRng

SLOT_S = 3600


@dataclass(frozen=True)
class Event:
    start: float
    end: float
    params: dict[str, Any] = field(default_factory=dict, compare=False)


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


class Continuous(Schedule):
    @override
    def active_fraction(self, t: np.ndarray, step_s: float) -> np.ndarray:
        return np.ones(np.shape(t))

    @override
    def envelope(self, t: np.ndarray) -> np.ndarray:
        return np.ones(np.shape(t))


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


class _EventSchedule(Schedule):
    is_event = True

    def __init__(self, rng: KeyedRng, source_id: str, emitter_id: str, duration: DurationDist) -> None:
        self.rng, self.source_id, self.emitter_id, self.duration = rng, source_id, emitter_id, duration
        self._slots: dict[int, list[Event]] = {}

    @abstractmethod
    def _slot_events(self, g: np.random.Generator, slot: int) -> list[Event]: ...

    def slot(self, h: int) -> list[Event]:
        ev = self._slots.get(h)
        if ev is None:
            ev = self._slot_events(self.rng.stream(self.source_id, self.emitter_id, "slot", h), h)
            self._slots[h] = ev
        return ev

    @override
    def events(self, t0: float, t1: float) -> list[Event]:
        """Events overlapping [t0, t1), ordered by start."""
        h0 = math.floor((t0 - self.duration.upper_s) / SLOT_S)
        out = [e for h in range(h0, math.floor(t1 / SLOT_S) + 1) for e in self.slot(h) if e.end > t0 and e.start < t1]
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
    def envelope(self, t: np.ndarray) -> np.ndarray:
        """1 inside an event, else 0 (burst ramps belong to the signal, from M6/M8)."""
        t = np.asarray(t, dtype=np.float64)
        out = np.zeros(t.shape)
        if t.size:
            for e in self.events(float(t.min()), float(t.max()) + 1.0):
                out[(t >= e.start) & (t < e.end)] = 1.0
        return out


class ExpGaps(_EventSchedule):
    def __init__(
        self,
        rng: KeyedRng,
        source_id: str,
        emitter_id: str,
        duration: DurationDist,
        mean_gap_s: float = 180.0,
        min_gap_s: float = 20.0,
    ) -> None:
        if not 0 <= min_gap_s <= mean_gap_s:
            raise ValueError("exp_gaps needs 0 <= min_gap_s <= mean_gap_s")
        if duration.upper_s > min_gap_s:
            raise ValueError("exp_gaps: event durations (max_s) must not exceed min_gap_s, or slots could overlap")
        super().__init__(rng, source_id, emitter_id, duration)
        self.mean_gap_s, self.min_gap_s = mean_gap_s, min_gap_s

    def _gap(self, g: np.random.Generator) -> float:
        return self.min_gap_s + float(g.exponential(self.mean_gap_s - self.min_gap_s))

    @override
    def _slot_events(self, g: np.random.Generator, slot: int) -> list[Event]:
        end_slot = (slot + 1) * SLOT_S
        t = slot * SLOT_S + self._gap(g)
        out = []
        while t < end_slot:
            d = float(self.duration.draw(g, 1)[0])
            out.append(Event(t, t + d, {"duration_s": d}))
            t = t + d + self._gap(g)
        return out


class PoissonEvents(_EventSchedule):
    def __init__(
        self,
        rng: KeyedRng,
        source_id: str,
        emitter_id: str,
        duration: DurationDist,
        rate_per_hour: float | Sequence[float] = 1.0,
    ) -> None:
        if duration.upper_s > SLOT_S:
            raise ValueError("poisson_events: max_s must be <= 3600 s")
        super().__init__(rng, source_id, emitter_id, duration)
        self.rate_per_hour = rate_per_hour

    def rate(self, slot: int) -> float:
        r = self.rate_per_hour
        return float(r) if isinstance(r, int | float) else float(r[slot % 24])  # UTC hour of day

    @override
    def _slot_events(self, g: np.random.Generator, slot: int) -> list[Event]:
        n = int(g.poisson(self.rate(slot)))
        starts = np.sort(g.uniform(0.0, SLOT_S, n)) + slot * SLOT_S
        durs = self.duration.draw(g, n)
        return [Event(float(s), float(s + d), {"duration_s": float(d)}) for s, d in zip(starts, durs, strict=True)]


@dataclass(frozen=True)
class AmModel:
    """Tumbling: gain_dB(t) = -(depth_db / 2) (1 - cos(2 pi t / period_s + phase_rad)), t in absolute Unix seconds."""

    period_s: float
    depth_db: float
    phase_rad: float = 0.0

    def gain_db(self, t: np.ndarray) -> np.ndarray:
        frac = np.fmod(np.asarray(t, dtype=np.float64), self.period_s) / self.period_s  # exact reduction first
        return -(self.depth_db / 2) * (1 - np.cos(2 * np.pi * frac + self.phase_rad))
