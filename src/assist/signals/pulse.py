"""Shared parts of the pulse-shaped signals (`oqpsk`, `oqpsk_burst`, `sdpsk`).

Symbols sit on an absolute grid: symbol k is at Unix time k / rs (rs is an integer number of symbols per second),
and narrow sample n at n / rate with rate = sps * rs. The root-raised-cosine filter has span * sps + 1 taps and is
scaled so that unit-power symbols give unit mean power (sum of h^2 = sps).
"""

from functools import cache
from typing import override

import numpy as np
from pydantic import Field, field_validator

from assist.signals.base import Signal, SignalParams


def rrc_taps(alpha: float, sps: int, span: int) -> np.ndarray:
    """Root-raised-cosine taps (float64): span * sps + 1 of them, centred, sum of squares = sps."""
    t = np.arange(-(span * sps // 2), span * sps // 2 + 1, dtype=np.float64) / sps  # in symbols
    h = np.empty_like(t)
    a = alpha
    for i, x in enumerate(t):
        if x == 0:
            h[i] = 1 - a + 4 * a / np.pi
        elif a > 0 and abs(abs(x) - 1 / (4 * a)) < 1e-12:
            q = np.pi / (4 * a)
            h[i] = a / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(q) + (1 - 2 / np.pi) * np.cos(q))
        else:
            h[i] = (np.sin(np.pi * x * (1 - a)) + 4 * a * x * np.cos(np.pi * x * (1 + a))) / (
                np.pi * x * (1 - (4 * a * x) ** 2)
            )
    return h * np.sqrt(sps / np.sum(h**2))


@cache
def _taps32(alpha: float, sps: int, span: int) -> np.ndarray:
    h = rrc_taps(alpha, sps, span).astype(np.float32)
    h.flags.writeable = False
    return h


class ShapedParams(SignalParams):
    rs: int = Field(gt=0, description="Symbol rate (symbols per second, an integer).")
    alpha: float = Field(gt=0, le=1, description="Roll-off of the root-raised-cosine pulse.")
    sps: int = Field(ge=2, description="Samples per symbol at the preferred narrow rate.")
    span: int = Field(10, ge=2, description="Filter length in symbols (even).")

    @field_validator("span")
    @classmethod
    def _span(cls, v: int) -> int:
        if v % 2:
            raise ValueError("span must be even")
        return v


class ShapedSignal(Signal):
    """A linearly modulated signal: symbols on the absolute grid through the RRC filter (`shape` kernel)."""

    q_delay = False  # OQPSK: Q is delayed by half a symbol

    @property
    def sp(self) -> ShapedParams:
        p = self.p
        assert isinstance(p, ShapedParams)
        return p

    @override
    def bandwidth_hz(self) -> float:
        return self.sp.rs * (1 + self.sp.alpha)

    @override
    def preferred_rate(self) -> int:
        return self.sp.sps * self.sp.rs

    @override
    def rate_step(self) -> int:
        return 2 * self.sp.rs if self.q_delay else self.sp.rs

    def samples_per_symbol(self, rate: int) -> int:
        sps, rem = divmod(rate, self.sp.rs)
        if rem or sps < 2 or (self.q_delay and sps % 2):
            raise ValueError(f"{self.kind}: narrow rate {rate} Hz is not an allowed multiple of rs {self.sp.rs}")
        return sps

    def taps(self, sps: int) -> np.ndarray:
        return _taps32(self.sp.alpha, sps, self.sp.span)

    def symbol_range(self, n0: int, n: int, sps: int) -> tuple[int, int]:
        """[k_lo, k_hi): every symbol with a tap on samples [n0, n0 + n)."""
        c = (self.sp.span * sps) // 2
        qd = sps // 2 if self.q_delay else 0
        return (n0 - qd - c) // sps, (n0 + n - 1 + c) // sps + 1

    def shaped(self, si: np.ndarray, sq: np.ndarray, k_lo: int, n0: int, n: int, sps: int) -> np.ndarray:
        qd = sps // 2 if self.q_delay else 0
        return self.dsp.shape(si, sq, k_lo, self.taps(sps), sps, qd, n0, n)


def unpack(data: np.ndarray) -> np.ndarray:
    """Bytes -> bits (MSB first) as uint8 0/1."""
    return np.unpackbits(np.asarray(data, dtype=np.uint8))
