"""Signal kinds (plan §3.4, §5).

M2 knows every kind by name so scenarios can be written now. A kind's parameters are checked, and it can be
rendered, from the milestone that implements it.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class SignalKind:
    name: str
    mode: Literal["continuous", "event"]
    milestone: str


KINDS = {
    k.name: k
    for k in (
        SignalKind("cw", "continuous", "M2"),
        SignalKind("oqpsk", "continuous", "M6"),
        SignalKind("oqpsk_burst", "continuous", "M6"),
        SignalKind("sdpsk", "continuous", "M6"),
        SignalKind("lora", "event", "M6"),
        SignalKind("d8psk", "event", "M8"),
        SignalKind("am", "event", "M8"),
        SignalKind("noise_band", "continuous", "M8"),
        SignalKind("bank_clip", "event", "M8"),
    )
}
IMPLEMENTED = {"cw"}


def get_kind(name: str) -> SignalKind:
    if name not in KINDS:
        raise ValueError(f"unknown signal kind {name!r} (known: {', '.join(KINDS)})")
    return KINDS[name]


class Signal(ABC):
    """How an emitter is modulated. Output: complex64 with unit mean power while on (plan §3.4)."""

    kind: ClassVar[str]

    @abstractmethod
    def bandwidth_hz(self) -> float:
        """Occupied bandwidth (Hz) of the baseband signal, used by the narrow-rate rule (plan §4.2)."""

    @abstractmethod
    def preferred_rate(self) -> int:
        """Lowest narrow rate (Hz, a multiple of 100) the signal wants, e.g. sps * symbol rate."""

    @abstractmethod
    def render(self, n0: int, n: int, rate: int) -> np.ndarray:
        """n samples (complex64, unit mean power while on) at `rate`, starting at absolute narrow index n0.

        Index n is at Unix time n / rate, so content is addressed by absolute time (plan §3.1).
        """


def make_signal(kind: str, params: dict[str, object]) -> Signal:
    k = get_kind(kind)
    if kind not in IMPLEMENTED:
        raise NotImplementedError(f"signal kind {kind!r} is rendered from milestone {k.milestone}")
    from assist.signals.cw import CW

    if params:
        raise ValueError(f"cw takes no parameters, got {sorted(params)}")
    return CW()
