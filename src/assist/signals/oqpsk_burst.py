"""`oqpsk_burst` (plan §5, §6.4 `periodic_bursts`): the ORBCOMM gateway downlink, OQPSK bursts at 28.8 ksym/s.

Segment s starts at s * period_s (absolute Unix time) and holds `blocks` blocks (4 or 5, drawn per segment) of
block_ms each, gap_ms apart: block b of segment s covers [s period + b (block + gap), ... + block). Each block
holds random OQPSK symbols (I, Q = +-1/sqrt(2), Q half a symbol late, RRC alpha 0.6) and raised-cosine ramps of
ramp_ms centred on its edges, so a block lasts block_ms between half-amplitude points. With the defaults a
segment is on for 257.5 or 322.5 ms and off for 692.5 or 627.5 ms.

Block edges sit on the absolute symbol grid (period, block and gap are whole numbers of symbols). The block count
comes from `random_bytes` keyed (source, emitter, "segment") with counter s, the symbols from (source, emitter,
"block") with counter 16 s + b.
"""

import math
from typing import TYPE_CHECKING, override

import numpy as np
from pydantic import Field, model_validator

from assist.signals.pulse import ShapedParams, ShapedSignal, unpack

if TYPE_CHECKING:
    from types import ModuleType

    from assist.signals.base import Keys, SignalParams

_AMP = np.float32(1 / np.sqrt(2))
MAX_BLOCKS = 16


def _whole(x: float, what: str) -> int:
    k = round(x)
    if abs(x - k) > 1e-6:
        raise ValueError(f"{what} must be a whole number of symbols (got {x:g})")
    return k


class OqpskBurstParams(ShapedParams):
    rs: int = Field(28_800, gt=0)
    alpha: float = Field(0.6, gt=0, le=1)
    sps: int = Field(4, ge=2)
    blocks: int | tuple[int, int] = Field((4, 5), description="Blocks per segment: a number, or [lo, hi] drawn.")
    block_ms: float = Field(62.5, gt=0)
    gap_ms: float = Field(2.5, ge=0)
    period_s: float = Field(0.95, gt=0)
    ramp_ms: float = Field(0.5, ge=0)

    @model_validator(mode="after")
    def _check(self) -> OqpskBurstParams:
        lo, hi = self.block_range
        if not 1 <= lo <= hi <= MAX_BLOCKS:
            raise ValueError(f"blocks must be in 1 .. {MAX_BLOCKS} (lo <= hi)")
        _whole(self.period_s * self.rs, "period_s")
        _whole(self.block_ms * self.rs / 1000, "block_ms")
        _whole(self.gap_ms * self.rs / 1000, "gap_ms")
        on = hi * self.block_ms + (hi - 1) * self.gap_ms + self.ramp_ms
        if on >= self.period_s * 1000:
            raise ValueError("a segment's blocks do not fit in period_s")
        if self.ramp_ms >= min(self.block_ms, self.gap_ms or self.block_ms):
            raise ValueError("ramp_ms must be shorter than block_ms and gap_ms")
        return self

    @property
    def block_range(self) -> tuple[int, int]:
        return (self.blocks, self.blocks) if isinstance(self.blocks, int) else self.blocks


class OQPSKBurst(ShapedSignal):
    kind = "oqpsk_burst"
    Params = OqpskBurstParams
    q_delay = True

    def __init__(self, params: SignalParams, keys: Keys, dsp: ModuleType) -> None:
        super().__init__(params, keys, dsp)
        p = self.bp
        self.period = _whole(p.period_s * p.rs, "period_s")  # in symbols
        self.block = _whole(p.block_ms * p.rs / 1000, "block_ms")
        self.gap = _whole(p.gap_ms * p.rs / 1000, "gap_ms")
        self._seg_key = keys.key64("segment")
        self._blk_key = keys.key64("block")

    @property
    def bp(self) -> OqpskBurstParams:
        p = self.p
        assert isinstance(p, OqpskBurstParams)
        return p

    def n_blocks(self, s: int) -> int:
        lo, hi = self.bp.block_range
        if lo == hi:
            return lo
        return lo + int(self.dsp.random_bytes(self._seg_key, s, 1)[0]) % (hi - lo + 1)

    def blocks(self, k0: int, k1: int) -> list[tuple[int, int, int]]:
        """(segment, block, first symbol) of every block whose symbols (with ramps) may touch [k0, k1)."""
        pad = self.block + self.gap
        out = []
        for s in range((k0 - pad) // self.period, k1 // self.period + 1):
            for b in range(self.n_blocks(s)):
                a = s * self.period + b * (self.block + self.gap)
                if a - pad < k1 and a + self.block + pad > k0:
                    out.append((s, b, a))
        return out

    def on_intervals(self, t0: float, t1: float) -> list[tuple[float, float]]:
        """Blocks overlapping [t0, t1) between half-amplitude points (Unix seconds), for tests and truth."""
        rs = self.bp.rs
        k0, k1 = math.floor(t0 * rs), math.ceil(t1 * rs) + 1
        return [
            (a / rs, (a + self.block) / rs)
            for _, _, a in self.blocks(k0, k1)
            if (a + self.block) / rs > t0 and a / rs < t1
        ]

    @override
    def render(self, n0: int, n: int, rate: int) -> np.ndarray:
        sps = self.samples_per_symbol(rate)
        k_lo, k_hi = self.symbol_range(n0, n, sps)
        si = np.zeros(k_hi - k_lo, dtype=np.float32)
        sq = np.zeros(k_hi - k_lo, dtype=np.float32)
        env = np.zeros(n, dtype=np.float64)
        ramp = self.bp.ramp_ms * rate / 1000  # samples
        r_sym = math.ceil(self.bp.ramp_ms * self.bp.rs / 2000) + 1  # symbols needed beyond each edge
        idx = np.arange(n0, n0 + n, dtype=np.int64)
        for s, b, a in self.blocks(k_lo, k_hi):
            first, last = a - r_sym, a + self.block + r_sym  # symbols [first, last)
            bits = unpack(self.dsp.random_bytes(self._blk_key, MAX_BLOCKS * s + b, math.ceil(2 * (last - first) / 8)))
            sym = (1 - 2 * bits[: 2 * (last - first)].astype(np.float32)) * _AMP
            lo, hi = max(first, k_lo), min(last, k_hi)
            if lo < hi:
                si[lo - k_lo : hi - k_lo] = sym[0::2][lo - first : hi - first]
                sq[lo - k_lo : hi - k_lo] = sym[1::2][lo - first : hi - first]
            j0 = max(0, math.floor(a * sps - ramp / 2) - n0)  # the envelope is 0 outside the block and its ramps
            j1 = min(n, math.ceil((a + self.block) * sps + ramp / 2) + 1 - n0)
            if j0 < j1:
                env[j0:j1] += _block_envelope(idx[j0:j1], a * sps, (a + self.block) * sps, ramp)
        if not env.any():
            return np.zeros(n, dtype=np.complex64)
        return (self.shaped(si, sq, k_lo, n0, n, sps) * env).astype(np.complex64)


def _block_envelope(idx: np.ndarray, start: int, end: int, ramp: float) -> np.ndarray:
    """1 between start and end (half-amplitude points, samples), raised-cosine ramps of `ramp` samples centred on
    the edges, 0 outside."""
    x = idx.astype(np.float64)
    if ramp <= 0:
        return ((x >= start) & (x < end)).astype(np.float64)
    up = np.clip((x - (start - ramp / 2)) / ramp, 0.0, 1.0)
    dn = np.clip(((end + ramp / 2) - x) / ramp, 0.0, 1.0)
    return 0.5 * (1 - np.cos(np.pi * up)) * 0.5 * (1 - np.cos(np.pi * dn))
