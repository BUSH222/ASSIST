"""`d8psk` (plan §5): VDL Mode 2 bursts, 10.5 ksym/s differential 8-PSK, an event signal (M7).

A burst is a whole number of symbols on the absolute symbol grid (symbol k at Unix time k / rs; the first symbol is
the first grid point at or after the event start):
- `ramp_up` symbols (phase change 0) whose amplitude rises as a raised cosine;
- the 16-symbol VDL2 synchronisation sequence;
- `header` + payload symbols with random phase changes (Gray-coded tribits, from `random_bytes` keyed
  (source, emitter, "burst") with the burst start in whole microseconds as counter);
- `ramp_down` symbols (phase change 0) whose amplitude falls.
The symbols go through a raised-cosine filter (alpha 0.6, as VDL2 transmitters use) by the `shape` kernel. The burst
length is drawn per burst (`burst_ms`, a number or an inclusive [lo, hi] range, rounded to whole symbols). Not
decodable: no scrambler, FEC, interleaving or AVLC framing. Unit mean power while on (ramps aside).
"""

import math
from functools import cache
from typing import TYPE_CHECKING, Any, override

import numpy as np
from pydantic import Field, model_validator

from assist.signals.base import Signal, SignalParams, float_range

if TYPE_CHECKING:
    from types import ModuleType

    from assist.emitters.schedules import Event
    from assist.signals.base import Keys

US = 1_000_000
# Phase change (multiples of pi/4) per tribit, Gray coded (VDL2: 000 -> 0, 001 -> 1, 011 -> 2, 010 -> 3, ...).
GRAY = np.array([0, 1, 3, 2, 7, 6, 4, 5], dtype=np.int64)
# VDL2 synchronisation sequence (16 tribits).
SYNC = (0b000, 0b010, 0b011, 0b110, 0b000, 0b001, 0b101, 0b110, 0b001, 0b100, 0b011, 0b111, 0b101, 0b111, 0b100, 0b010)


class D8pskParams(SignalParams):
    rs: int = Field(10_500, gt=0, description="Symbol rate (symbols per second).")
    alpha: float = Field(0.6, gt=0, le=1, description="Roll-off of the raised-cosine pulse.")
    sps: int = Field(4, ge=2, description="Samples per symbol at the preferred narrow rate.")
    span: int = Field(8, ge=2, description="Filter length in symbols (even).")
    ramp_up: int = Field(5, ge=1)
    header: int = Field(17, ge=0)
    ramp_down: int = Field(3, ge=1)
    burst_ms: float | tuple[float, float] = Field(
        (20.0, 23.0), description="Whole burst length per burst (ms): a number or an inclusive [lo, hi] range."
    )

    @model_validator(mode="after")
    def _check(self) -> D8pskParams:
        if self.span % 2:
            raise ValueError("span must be even")
        lo, hi = float_range(self.burst_ms)
        if not 0 < lo <= hi:
            raise ValueError("burst_ms must be > 0 (lo <= hi)")
        if math.floor(hi * self.rs / 1000) < self.fixed_symbols:
            raise ValueError(f"burst_ms must allow at least {self.fixed_symbols} symbols (ramps, sync, header)")
        return self

    @property
    def fixed_symbols(self) -> int:
        return self.ramp_up + len(SYNC) + self.header + self.ramp_down

    def symbol_range(self) -> tuple[int, int]:
        lo, hi = float_range(self.burst_ms)
        return max(self.fixed_symbols, math.ceil(lo * self.rs / 1000 - 1e-9)), math.floor(hi * self.rs / 1000 + 1e-9)


def rc_taps(alpha: float, sps: int, span: int) -> np.ndarray:
    """Raised-cosine taps (float64): span * sps + 1 of them, centred, sum of squares = sps (unit-power symbols give
    unit mean power)."""
    t = np.arange(-(span * sps // 2), span * sps // 2 + 1, dtype=np.float64) / sps  # in symbols
    h = np.sinc(t)
    d = 1 - (2 * alpha * t) ** 2
    sing = np.abs(d) < 1e-12
    h[~sing] *= np.cos(np.pi * alpha * t[~sing]) / d[~sing]
    h[sing] = np.pi / 4 * np.sinc(1 / (2 * alpha))
    return h * np.sqrt(sps / np.sum(h**2))


@cache
def _taps32(alpha: float, sps: int, span: int) -> np.ndarray:
    h = rc_taps(alpha, sps, span).astype(np.float32)
    h.flags.writeable = False
    return h


def _ramp(n: int) -> np.ndarray:
    """Raised-cosine amplitude rise over n symbols (exclusive of 0 and 1)."""
    return 0.5 * (1 - np.cos(np.pi * (np.arange(n) + 0.5) / n))


class D8PSK(Signal):
    kind = "d8psk"
    Params = D8pskParams
    mode = "event"

    def __init__(self, params: SignalParams, keys: Keys, dsp: ModuleType) -> None:
        super().__init__(params, keys, dsp)
        self._key = keys.key64("burst")

    @property
    def dp(self) -> D8pskParams:
        p = self.p
        assert isinstance(p, D8pskParams)
        return p

    @override
    def bandwidth_hz(self) -> float:
        return self.dp.rs * (1 + self.dp.alpha)

    @override
    def preferred_rate(self) -> int:
        return self.dp.sps * self.dp.rs

    @override
    def rate_step(self) -> int:
        return self.dp.rs

    @override
    @classmethod
    def draw_event(cls, p: SignalParams, g: np.random.Generator) -> tuple[float, dict[str, Any]]:
        assert isinstance(p, D8pskParams)
        lo, hi = p.symbol_range()
        n = int(g.integers(lo, hi, endpoint=True))
        d = n / p.rs
        return d, {"duration_s": d, "symbols": n}

    @override
    @classmethod
    def max_duration_s(cls, p: SignalParams) -> float:
        assert isinstance(p, D8pskParams)
        return (p.symbol_range()[1] + 1) / p.rs  # + 1: the first symbol may sit up to one symbol after the start

    def first_symbol(self, ev: Event) -> int:
        """Absolute index of the burst's first symbol: the first grid point at or after the event start."""
        s_us = round(ev.start * US)
        return -((-s_us * self.dp.rs) // US)

    def symbols(self, ev: Event) -> np.ndarray:
        """The burst's complex symbols (amplitude ramps applied)."""
        p = self.dp
        n = int(ev.params["symbols"])
        n_rand = n - p.ramp_up - len(SYNC) - p.ramp_down
        tri = np.zeros(n, dtype=np.int64)
        tri[p.ramp_up : p.ramp_up + len(SYNC)] = SYNC
        raw = self.dsp.random_bytes(self._key, round(ev.start * US), n_rand)
        tri[p.ramp_up + len(SYNC) : p.ramp_up + len(SYNC) + n_rand] = raw.astype(np.int64) & 7
        steps = GRAY[tri]
        steps[: p.ramp_up] = 0
        steps[n - p.ramp_down :] = 0
        phase = np.cumsum(steps) % 8 * (np.pi / 4)
        amp = np.ones(n)
        amp[: p.ramp_up] = _ramp(p.ramp_up)
        amp[n - p.ramp_down :] = _ramp(p.ramp_down)[::-1]
        return amp * np.exp(1j * phase)

    @override
    def event_bounds(self, ev: Event) -> tuple[float, float]:
        p = self.dp
        k0 = self.first_symbol(ev)
        half = p.span // 2
        return (k0 - half) / p.rs, (k0 + int(ev.params["symbols"]) + half) / p.rs

    @override
    def render_events(self, events: list[Event], n0: int, n: int, rate: int) -> np.ndarray:
        p = self.dp
        sps, rem = divmod(rate, p.rs)
        if rem or sps < 2:
            raise ValueError(f"d8psk: narrow rate {rate} Hz is not a multiple (>= 2) of rs {p.rs}")
        out = np.zeros(n, dtype=np.complex64)
        h = _taps32(p.alpha, sps, p.span)
        half = p.span // 2
        for ev in events:
            k0 = self.first_symbol(ev)
            ns = int(ev.params["symbols"])
            a = max(n0, (k0 - half) * sps)  # samples the burst touches (filter tails included)
            b = min(n0 + n, (k0 + ns + half) * sps + 1)
            if b <= a:
                continue
            s = self.symbols(ev)
            out[a - n0 : b - n0] += self.dsp.shape(s.real, s.imag, k0, h, sps, 0, a, b - a)
        return out
