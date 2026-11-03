"""`cw`: an unmodulated carrier (plan §5). Frequency, level and on/off come from the emitter."""

from typing import override

import numpy as np

from assist.signals.base import Signal


class CW(Signal):
    kind = "cw"

    @override
    def bandwidth_hz(self) -> float:
        return 0.0

    @override
    def render(self, n0: int, n: int, rate: int) -> np.ndarray:
        return np.ones(n, dtype=np.complex64)
