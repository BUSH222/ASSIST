"""`sdpsk`: ORBCOMM subscriber downlink, symmetric differential PSK at 4800 sym/s, RRC alpha 0.4.

The absolute phase state of symbol k is p_k = (k mod 2) + 2 bit(k), phase pi/2 p_k: every step is +-90 deg (the
step's sign carries the data), and any symbol can be computed from its index alone. Bits are random, 4096 per
block from `random_bytes` keyed (source, emitter, "sym") with the block index (k div 4096) as counter.
"""

from collections import OrderedDict
from typing import TYPE_CHECKING, override

import numpy as np
from pydantic import Field

from assist.signals.base import register_signal
from assist.signals.pulse import ShapedParams, ShapedSignal, unpack

if TYPE_CHECKING:
    from types import ModuleType

    from assist.signals.base import Keys, SignalParams

BLOCK = 4096
_RE = np.array([1, 0, -1, 0], dtype=np.float32)  # j^p
_IM = np.array([0, 1, 0, -1], dtype=np.float32)


class SdpskParams(ShapedParams):
    rs: int = Field(4800, gt=0)
    alpha: float = Field(0.4, gt=0, le=1)
    sps: int = Field(8, ge=2)


@register_signal("sdpsk")
class SDPSK(ShapedSignal):
    Params = SdpskParams

    def __init__(self, params: SignalParams, keys: Keys, dsp: ModuleType) -> None:
        super().__init__(params, keys, dsp)
        self._key = keys.key64("sym")
        self._bits: OrderedDict[int, np.ndarray] = OrderedDict()

    def bits(self, k_lo: int, k_hi: int) -> np.ndarray:
        """Data bits of absolute symbols [k_lo, k_hi)."""
        b0, b1 = k_lo // BLOCK, (k_hi - 1) // BLOCK
        parts = []
        for b in range(b0, b1 + 1):
            hit = self._bits.get(b)
            if hit is None:
                hit = unpack(self.dsp.random_bytes(self._key, b, BLOCK // 8))
                self._bits[b] = hit
                while len(self._bits) > 8:
                    self._bits.popitem(last=False)
            parts.append(hit)
        return np.concatenate(parts)[k_lo - b0 * BLOCK : k_hi - b0 * BLOCK]

    def states(self, k_lo: int, k_hi: int) -> np.ndarray:
        """Phase states p_k = (k mod 2) + 2 bit(k), in quarter turns."""
        k = np.arange(k_lo, k_hi, dtype=np.int64)
        return (k % 2 + 2 * self.bits(k_lo, k_hi)).astype(np.int64)

    @override
    def render(self, n0: int, n: int, rate: int) -> np.ndarray:
        sps = self.samples_per_symbol(rate)
        k_lo, k_hi = self.symbol_range(n0, n, sps)
        p = self.states(k_lo, k_hi)
        return self.shaped(_RE[p], _IM[p], k_lo, n0, n, sps)
