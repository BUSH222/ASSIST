"""`gfsk` (M7): VDL Mode 4 style bursts, 19.2 kbit/s Gaussian FSK (modulation index 0.25, BT 0.28), an event signal.

M7 found that the bursts on 136.925 MHz (one of the VDL Mode 4 signalling channels) are not D8PSK: their envelope is
constant and their instantaneous frequency swings +-2.4 kHz (= h rb / 2), with a centre-peaked spectrum; the bursts
on 136.975 MHz are VDL Mode 2 D8PSK (`d8psk`).

A burst is a whole number of bits on the absolute bit grid (bit k at Unix time k / rb; the first bit is the first
grid point at or after the event start). The bits are random (`random_bytes` keyed (source, emitter, "burst") with
the burst start in whole microseconds as counter): each bit differs from the previous one with probability
`transition_p` (0.5: random data; VDL4's NRZI coding makes transitions more frequent, which narrows the spectrum:
the recording's 136.925 MHz bursts fit ~0.65). The NRZ frequency pulse train goes through a Gaussian filter
(BT, `span` bits long) and is integrated into phase (phase 0 at the burst's first sample), so the samples of a burst
do not depend on the window. Raised-cosine amplitude ramps of `ramp` bits start and end the burst. Constant envelope,
unit power while on (ramps aside). Not decodable: no training sequence, CRC or framing.
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


class GfskParams(SignalParams):
    rb: int = Field(19_200, gt=0, description="Bit rate (bits per second).")
    h: float = Field(0.25, gt=0, le=1, description="Modulation index: peak deviation h rb / 2.")
    bt: float = Field(0.28, gt=0, description="Bandwidth-time product of the Gaussian frequency filter.")
    sps: int = Field(4, ge=2, description="Samples per bit at the preferred narrow rate.")
    span: int = Field(4, ge=1, description="Gaussian filter length in bits.")
    ramp: int = Field(4, ge=1, description="Ramp-up and ramp-down length in bits.")
    transition_p: float = Field(
        0.5, gt=0, lt=1, description="Probability that a bit differs from the previous one (0.5: random data)."
    )
    burst_ms: float | tuple[float, float] = Field(
        (20.5, 22.0), description="Whole burst length per burst (ms): a number or an inclusive [lo, hi] range."
    )

    @model_validator(mode="after")
    def _check(self) -> GfskParams:
        lo, hi = float_range(self.burst_ms)
        if not 0 < lo <= hi:
            raise ValueError("burst_ms must be > 0 (lo <= hi)")
        if math.floor(hi * self.rb / 1000) < 2 * self.ramp + 1:
            raise ValueError(f"burst_ms must allow more than {2 * self.ramp} bits (the ramps)")
        return self

    def bit_range(self) -> tuple[int, int]:
        lo, hi = float_range(self.burst_ms)
        return max(2 * self.ramp + 1, math.ceil(lo * self.rb / 1000 - 1e-9)), math.floor(hi * self.rb / 1000 + 1e-9)


@cache
def gauss_pulse(bt: float, sps: int, span: int) -> np.ndarray:
    """Gaussian frequency filter (float64, span * sps + 1 taps, sum 1): sigma = sqrt(ln 2) / (2 pi BT) bits."""
    sigma = math.sqrt(math.log(2)) / (2 * math.pi * bt) * sps
    t = np.arange(-(span * sps // 2), span * sps // 2 + 1, dtype=np.float64)
    g = np.exp(-0.5 * (t / sigma) ** 2)
    g /= g.sum()
    g.flags.writeable = False
    return g


def _ramp(n: int) -> np.ndarray:
    return 0.5 * (1 - np.cos(np.pi * (np.arange(n) + 0.5) / n))


class GFSK(Signal):
    kind = "gfsk"
    Params = GfskParams
    mode = "event"

    def __init__(self, params: SignalParams, keys: Keys, dsp: ModuleType) -> None:
        super().__init__(params, keys, dsp)
        self._key = keys.key64("burst")

    @property
    def gp(self) -> GfskParams:
        p = self.p
        assert isinstance(p, GfskParams)
        return p

    @override
    def bandwidth_hz(self) -> float:
        p = self.gp
        return p.h * p.rb + p.rb  # Carson: 2 (deviation + rb / 2)

    @override
    def preferred_rate(self) -> int:
        return self.gp.sps * self.gp.rb

    @override
    def rate_step(self) -> int:
        return self.gp.rb

    @override
    @classmethod
    def draw_event(cls, p: SignalParams, g: np.random.Generator) -> tuple[float, dict[str, Any]]:
        assert isinstance(p, GfskParams)
        lo, hi = p.bit_range()
        n = int(g.integers(lo, hi, endpoint=True))
        d = n / p.rb
        return d, {"duration_s": d, "bits": n}

    @override
    @classmethod
    def max_duration_s(cls, p: SignalParams) -> float:
        assert isinstance(p, GfskParams)
        return (p.bit_range()[1] + 1) / p.rb

    def first_bit(self, ev: Event) -> int:
        s_us = round(ev.start * US)
        return -((-s_us * self.gp.rb) // US)

    def burst(self, ev: Event, sps: int) -> np.ndarray:
        """The whole burst at sps samples per bit (complex128), from its first bit."""
        p = self.gp
        n = int(ev.params["bits"])
        u = self.dsp.random_bytes(self._key, round(ev.start * US), 2 * n).view("<u2")
        flips = np.cumsum(u < round(p.transition_p * 65536))  # transitions, from 16-bit uniforms
        bits = np.where(flips % 2 == 0, 1.0, -1.0)
        freq = np.convolve(np.repeat(bits, sps), gauss_pulse(p.bt, sps, p.span), mode="same") * (p.h * p.rb / 2)
        phase = 2 * np.pi * np.cumsum(freq) / (sps * p.rb)
        amp = np.ones(n * sps)
        r = p.ramp * sps
        amp[:r] = _ramp(r)
        amp[-r:] = _ramp(r)[::-1]
        return amp * np.exp(1j * phase)

    @override
    def event_bounds(self, ev: Event) -> tuple[float, float]:
        k0 = self.first_bit(ev)
        return k0 / self.gp.rb, (k0 + int(ev.params["bits"])) / self.gp.rb

    @override
    def render_events(self, events: list[Event], n0: int, n: int, rate: int) -> np.ndarray:
        p = self.gp
        sps, rem = divmod(rate, p.rb)
        if rem or sps < 2:
            raise ValueError(f"gfsk: narrow rate {rate} Hz is not a multiple (>= 2) of rb {p.rb}")
        out = np.zeros(n, dtype=np.complex64)
        for ev in events:
            a0 = self.first_bit(ev) * sps
            a1 = a0 + int(ev.params["bits"]) * sps
            a, b = max(n0, a0), min(n0 + n, a1)
            if b <= a:
                continue
            out[a - n0 : b - n0] += self.burst(ev, sps)[a - a0 : b - a0].astype(np.complex64)
        return out
