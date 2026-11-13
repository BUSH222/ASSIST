"""`pretty_gfsk`: bursts of 38.4 kbaud Gaussian FSK, like the downlink of PRETTY (NORAD 58023) on 137.78 MHz.

An external ASSIST module: this file is everything the kind needs. A scenario uses it by listing the file:

    plugins:
      - path: examples/plugins/pretty_gfsk.py
    sources:
      - id: pretty
        orbit: {norad: 58023}
        emitters:
          - id: tm
            signal: {kind: pretty_gfsk}
            freq_hz: 137.78e6
            schedule: {type: exp_gaps, mean_gap_s: 6, min_gap_s: 1}
            level: {cn0_zenith_dbhz: 62}

It is an *event* signal: the emitter's schedule places the bursts, `draw_event` draws each burst's length, and
`render_events` returns the samples of any window. Doppler, level, the antenna pattern and the pass itself come from
the engine; the signal only produces unit-power baseband samples at the narrow rate (a multiple of the bit rate).

A burst is a whole number of bits on the absolute bit grid (bit k at Unix time k / rb; the first bit is the first
grid point at or after the event start): a preamble of alternating bits, then random data. The NRZ frequency pulses
go through a Gaussian filter (BT) and are integrated into phase, starting from phase 0 at the burst's first sample,
so a burst's samples do not depend on the window they are rendered in. Raised-cosine amplitude ramps of `ramp` bits
start and end it.

Only the bit rate and the modulation type are taken from PRETTY. The modulation index, BT, preamble and burst
lengths below are placeholders (common values for GFSK packet radios), to be replaced by measured ones; there is no
framing, so the bursts are not decodable.
"""

import math
from functools import cache
from typing import Any, override

import numpy as np
from pydantic import Field, model_validator

from assist import api

__version__ = "0.1.0"
ASSIST_API = "1.0"  # the assist.api version this module was written for

US = 1_000_000


class PrettyGfskParams(api.SignalParams):
    rb: int = Field(38_400, gt=0, description="Bit rate (bits per second).")
    h: float = Field(0.5, gt=0, le=1, description="Modulation index: peak deviation h rb / 2.")
    bt: float = Field(0.5, gt=0, description="Bandwidth-time product of the Gaussian frequency filter.")
    sps: int = Field(4, ge=2, description="Samples per bit at the preferred narrow rate.")
    span: int = Field(4, ge=1, description="Gaussian filter length in bits.")
    ramp: int = Field(8, ge=1, description="Ramp-up and ramp-down length in bits.")
    preamble_bits: int = Field(64, ge=0, description="Alternating bits (0101...) at the start of every burst.")
    burst_ms: float | tuple[float, float] = Field(
        (150.0, 400.0), description="Burst length per burst (ms): a number or an inclusive [lo, hi] range."
    )

    @model_validator(mode="after")
    def _check(self) -> PrettyGfskParams:
        lo, hi = api.float_range(self.burst_ms)
        if not 0 < lo <= hi:
            raise ValueError("burst_ms must be > 0 (lo <= hi)")
        if self.bit_range()[1] < self.min_bits():
            raise ValueError(f"burst_ms must allow at least {self.min_bits()} bits (preamble and ramps)")
        return self

    def min_bits(self) -> int:
        return max(self.preamble_bits, 2 * self.ramp) + 1

    def bit_range(self) -> tuple[int, int]:
        """Fewest and most bits of a burst."""
        lo, hi = api.float_range(self.burst_ms)
        return max(self.min_bits(), math.ceil(lo * self.rb / 1000 - 1e-9)), math.floor(hi * self.rb / 1000 + 1e-9)


@cache
def gauss_taps(bt: float, sps: int, span: int) -> np.ndarray:
    """Gaussian frequency filter (float64, span * sps + 1 taps, sum 1): sigma = sqrt(ln 2) / (2 pi BT) bits."""
    sigma = math.sqrt(math.log(2)) / (2 * math.pi * bt) * sps
    t = np.arange(-(span * sps // 2), span * sps // 2 + 1, dtype=np.float64)
    g = np.exp(-0.5 * (t / sigma) ** 2)
    g /= g.sum()
    g.flags.writeable = False
    return g


@api.register_signal("pretty_gfsk")
class PrettyGfsk(api.Signal):
    Params = PrettyGfskParams
    mode = "event"

    def __init__(self, params: api.SignalParams, keys: api.Keys, dsp: Any) -> None:
        super().__init__(params, keys, dsp)
        # One 64-bit key for all data bits of this emitter; the counter (below) is the burst's start time.
        self._key = keys.key64("data")
        self._last: tuple[tuple[int, int], np.ndarray] | None = None  # the burst rendered last (bursts span chunks)

    @property
    def gp(self) -> PrettyGfskParams:
        p = self.p
        assert isinstance(p, PrettyGfskParams)
        return p

    # --- what the engine needs to pick the narrow rate ---

    @override
    def bandwidth_hz(self) -> float:
        p = self.gp
        return p.h * p.rb + p.rb  # Carson's rule: 2 (deviation + rb / 2)

    @override
    def preferred_rate(self) -> int:
        return self.gp.sps * self.gp.rb

    @override
    def rate_step(self) -> int:
        return self.gp.rb  # the narrow rate is a whole number of samples per bit

    # --- events ---

    @override
    @classmethod
    def draw_event(cls, p: api.SignalParams, g: np.random.Generator) -> tuple[float, dict[str, Any]]:
        """One burst's duration and parameters, from the schedule's stream for this burst."""
        assert isinstance(p, PrettyGfskParams)
        lo, hi = p.bit_range()
        n = int(g.integers(lo, hi, endpoint=True))
        d = n / p.rb
        return d, {"duration_s": d, "bits": n}

    @override
    @classmethod
    def max_duration_s(cls, p: api.SignalParams) -> float:
        assert isinstance(p, PrettyGfskParams)
        return (p.bit_range()[1] + 1) / p.rb

    def first_bit(self, ev: api.Event) -> int:
        """Absolute index of the burst's first bit: the first bit time at or after the event start."""
        return -((-round(ev.start * US) * self.gp.rb) // US)

    @override
    def event_bounds(self, ev: api.Event) -> tuple[float, float]:
        k0 = self.first_bit(ev)
        return k0 / self.gp.rb, (k0 + int(ev.params["bits"])) / self.gp.rb

    def bits(self, ev: api.Event) -> np.ndarray:
        """The burst's bits (0/1): the preamble, then random data addressed by the burst's start time."""
        p = self.gp
        n = int(ev.params["bits"])
        data = api.unpack(self.dsp.random_bytes(self._key, round(ev.start * US), (n + 7) // 8))[:n]
        npre = min(p.preamble_bits, n)
        data[:npre] = np.arange(npre) % 2
        return data

    def burst(self, ev: api.Event, sps: int) -> np.ndarray:
        """The whole burst at sps samples per bit (complex64), from its first bit."""
        ident = (round(ev.start * US), sps)
        if self._last is not None and self._last[0] == ident:
            return self._last[1]
        p = self.gp
        nrz = 2.0 * self.bits(ev) - 1.0
        freq = np.convolve(np.repeat(nrz, sps), gauss_taps(p.bt, sps, p.span), mode="same") * (p.h * p.rb / 2)
        phase = 2 * np.pi * np.cumsum(freq) / (sps * p.rb)
        amp = np.ones(len(phase))
        r = p.ramp * sps
        edge = 0.5 * (1 - np.cos(np.pi * (np.arange(r) + 0.5) / r))
        amp[:r] = edge
        amp[-r:] = edge[::-1]
        x = (amp * np.exp(1j * phase)).astype(np.complex64)
        self._last = (ident, x)
        return x

    @override
    def render_events(self, events: list[api.Event], n0: int, n: int, rate: int) -> np.ndarray:
        """The given bursts in absolute narrow samples [n0, n0 + n) at `rate`; 0 elsewhere."""
        sps, rem = divmod(rate, self.gp.rb)
        if rem or sps < 2:
            raise ValueError(f"pretty_gfsk: narrow rate {rate} Hz is not a multiple (>= 2) of rb {self.gp.rb}")
        out = np.zeros(n, dtype=np.complex64)
        for ev in events:
            a0 = self.first_bit(ev) * sps  # the burst's first sample, absolute
            a1 = a0 + int(ev.params["bits"]) * sps
            a, b = max(n0, a0), min(n0 + n, a1)
            if b > a:
                out[a - n0 : b - n0] += self.burst(ev, sps)[a - a0 : b - a0]
        return out
