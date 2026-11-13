"""`oqpsk`: METEOR LRPT, 72 ksym/s OQPSK with RRC alpha 0.6.

I and Q carry +-1/sqrt(2) symbols; Q is delayed by half a symbol. Framing:
- `meteor_cadu`: CADUs of 1024 bytes: the ASM 0x1ACFFC1D, then 1020 random bytes through the CCSDS randomiser. The
  K = 7 r = 1/2 convolutional code (171/133) turns a CADU's 8192 bits into 16384 coded bits, i.e. 8192 QPSK symbols
  (113.8 ms at 72k): frame f holds symbols [8192 f, 8192 (f + 1)). The encoder needs only the previous 6 input bits,
  the end of frame f - 1's randomised payload, so any frame is encoded on its own. The payload is random (no
  Reed-Solomon, no real instrument data): realistic at the spectrum and symbol level, not decodable.
- `random`: random symbols, 4096 per keyed block.
Payload bytes come from `random_bytes` keyed (source, emitter, "frame") with the frame index as counter
(`random`: (source, emitter, "sym") with the block index), so frames never repeat and any window is random-access.
"""

from collections import OrderedDict
from typing import TYPE_CHECKING, Literal, override

import numpy as np
from pydantic import Field

from assist.signals.base import register_signal
from assist.signals.pulse import ShapedParams, ShapedSignal, unpack

if TYPE_CHECKING:
    from types import ModuleType

    from assist.signals.base import Keys, SignalParams

ASM = np.frombuffer(bytes.fromhex("1ACFFC1D"), dtype=np.uint8)
CADU_BYTES = 1024
PAYLOAD_BYTES = CADU_BYTES - len(ASM)
FRAME_SYMBOLS = CADU_BYTES * 8  # rate 1/2 code, 2 coded bits per symbol
RANDOM_BLOCK = 4096
_CACHE = 24  # frames kept (a 1 s chunk spans ~9)
_AMP = np.float32(1 / np.sqrt(2))


class OqpskParams(ShapedParams):
    rs: int = Field(72_000, gt=0)
    alpha: float = Field(0.6, gt=0, le=1)
    sps: int = Field(4, ge=2)
    framing: Literal["meteor_cadu", "random"] = "meteor_cadu"


@register_signal("oqpsk")
class OQPSK(ShapedSignal):
    Params = OqpskParams
    q_delay = True

    def __init__(self, params: SignalParams, keys: Keys, dsp: ModuleType) -> None:
        super().__init__(params, keys, dsp)
        self._frames: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._payloads: OrderedDict[int, np.ndarray] = OrderedDict()
        p = self.p
        assert isinstance(p, OqpskParams)
        self.framing = p.framing
        self._key = self.keys.key64("frame" if p.framing == "meteor_cadu" else "sym")

    @override
    def render(self, n0: int, n: int, rate: int) -> np.ndarray:
        sps = self.samples_per_symbol(rate)
        k_lo, k_hi = self.symbol_range(n0, n, sps)
        si, sq = self.symbols(k_lo, k_hi)
        return self.shaped(si, sq, k_lo, n0, n, sps)

    def symbols(self, k_lo: int, k_hi: int) -> tuple[np.ndarray, np.ndarray]:
        """I and Q (float32, +-1/sqrt(2)) of absolute symbols [k_lo, k_hi)."""
        size = FRAME_SYMBOLS if self.framing == "meteor_cadu" else RANDOM_BLOCK
        f0, f1 = k_lo // size, (k_hi - 1) // size
        parts = [self._frame(f) for f in range(f0, f1 + 1)]
        si = np.concatenate([p[0] for p in parts])[k_lo - f0 * size : k_hi - f0 * size]
        sq = np.concatenate([p[1] for p in parts])[k_lo - f0 * size : k_hi - f0 * size]
        return si, sq

    def payload(self, f: int) -> np.ndarray:
        """Frame f's randomised payload (1020 bytes)."""
        d = self._payloads.get(f)
        if d is None:
            d = self.dsp.ccsds_randomise(self.dsp.random_bytes(self._key, f, PAYLOAD_BYTES))
            self._payloads[f] = d
            while len(self._payloads) > _CACHE + 1:
                self._payloads.popitem(last=False)
        return d

    def cadu(self, f: int) -> np.ndarray:
        return np.concatenate([ASM, self.payload(f)])

    def coded(self, f: int) -> np.ndarray:
        """Frame f's 16384 coded bits (0/1)."""
        state = int(self.payload(f - 1)[-1]) & 0x3F  # the previous frame's last 6 bits
        return self.dsp.conv_encode(unpack(self.cadu(f)), state)

    def _frame(self, f: int) -> tuple[np.ndarray, np.ndarray]:
        hit = self._frames.get(f)
        if hit is None:
            if self.framing == "meteor_cadu":
                bits = self.coded(f)
            else:
                bits = unpack(self.dsp.random_bytes(self._key, f, RANDOM_BLOCK // 4))
            sym = (1 - 2 * bits.astype(np.float32)) * _AMP
            hit = (np.ascontiguousarray(sym[0::2]), np.ascontiguousarray(sym[1::2]))
            self._frames[f] = hit
            while len(self._frames) > _CACHE:
                self._frames.popitem(last=False)
        return hit
