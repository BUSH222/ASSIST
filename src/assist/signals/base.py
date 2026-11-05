"""Signal kinds (plan §3.4, §5).

M2 knows every kind by name so scenarios can be written early. A kind's parameters are checked, and it can be
rendered, from the milestone that implements it. Parameters are a pydantic model per kind (unknown keys are
errors); validation fills in the defaults, so a resolved scenario lists every parameter.

Three modes:
- `continuous`: `render(n0, n, rate)` gives samples at any absolute narrow index (cw, oqpsk, oqpsk_burst, sdpsk,
  noise_band);
- `event`: the emitter's schedule places events; `draw_event` draws an event's parameters and duration (from the
  schedule's keyed slot stream; a duration of None means the schedule's duration distribution decides) and
  `render_events` renders the events overlapping a window (lora, d8psk, gfsk, am);
- `wideband` (M7): interference inside the receiver, rendered at the full sample rate in the time domain and added
  before the ADC, bypassing the synthesiser and the receiver response (noise_comb, the USB humps).

Random content (payload bits, frames, burst symbols) comes from `Keys`: 64-bit keys per (source, emitter, purpose)
for the counter-based `random_bytes` kernel, whose counter is an absolute index (frame, symbol chunk, segment, event
start), so any window is rendered on its own (plan §3.2).
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from types import ModuleType

    import numpy as np

    from assist.emitters.schedules import Event
    from assist.rng import KeyedRng


@dataclass(frozen=True)
class SignalKind:
    name: str
    mode: Literal["continuous", "event", "wideband"]
    milestone: str


KINDS = {
    k.name: k
    for k in (
        SignalKind("cw", "continuous", "M2"),
        SignalKind("oqpsk", "continuous", "M6"),
        SignalKind("oqpsk_burst", "continuous", "M6"),
        SignalKind("sdpsk", "continuous", "M6"),
        SignalKind("lora", "event", "M6"),
        SignalKind("d8psk", "event", "M7"),
        SignalKind("gfsk", "event", "M7"),
        SignalKind("am", "event", "M7"),
        SignalKind("noise_band", "continuous", "M7"),
        SignalKind("noise_comb", "wideband", "M7"),
        SignalKind("bank_clip", "event", "later (skipped in M7: everything synthetic for now)"),
    )
}
IMPLEMENTED = {"cw", "oqpsk", "oqpsk_burst", "sdpsk", "lora", "d8psk", "gfsk", "am", "noise_band", "noise_comb"}


def get_kind(name: str) -> SignalKind:
    if name not in KINDS:
        raise ValueError(f"unknown signal kind {name!r} (known: {', '.join(KINDS)})")
    return KINDS[name]


class SignalParams(BaseModel):
    """Base of every kind's parameter model: unknown keys are errors."""

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class Keys:
    """Random streams of one emitter: (source_id, emitter_id, purpose, ...) keys from the scenario seed."""

    rng: KeyedRng
    source_id: str
    emitter_id: str

    def key64(self, *parts: str | int) -> int:
        return self.rng.key64(self.source_id, self.emitter_id, *parts)


class Signal(ABC):
    """How an emitter is modulated. Output: complex64 with unit mean power while on (plan §3.4)."""

    kind: ClassVar[str]
    Params: ClassVar[type[SignalParams]]
    mode: ClassVar[Literal["continuous", "event", "wideband"]] = "continuous"
    constant: ClassVar[bool] = False  # the samples are all 1 (cw): the engine skips render()

    def __init__(self, params: SignalParams, keys: Keys, dsp: ModuleType) -> None:
        self.p, self.keys, self.dsp = params, keys, dsp

    @abstractmethod
    def bandwidth_hz(self) -> float:
        """Occupied bandwidth (Hz) of the baseband signal, used by the narrow-rate rule (plan §4.2)."""

    @abstractmethod
    def preferred_rate(self) -> int:
        """Lowest narrow rate (Hz, a multiple of rate_step) the signal wants, e.g. sps * symbol rate."""

    def rate_step(self) -> int:
        """The narrow rate must be a multiple of this (e.g. the symbol rate, so sps is an integer)."""
        return 100

    def render(self, n0: int, n: int, rate: int) -> np.ndarray:
        """Continuous signals: n samples (complex64) at `rate`, starting at absolute narrow index n0.

        Index n is at Unix time n / rate, so content is addressed by absolute time (plan §3.1).
        """
        raise NotImplementedError(f"{self.kind} is an event signal")

    # --- event signals ---

    @classmethod
    def draw_event(cls, p: SignalParams, g: np.random.Generator) -> tuple[float | None, dict[str, Any]]:
        """(duration in s, parameters) of one event, drawn from the schedule's keyed slot stream. A duration of None
        means the schedule's duration distribution decides (am, noise_comb)."""
        raise NotImplementedError(f"{cls.kind} is not an event signal")

    @classmethod
    def max_duration_s(cls, p: SignalParams) -> float | None:
        """Upper bound of an event's duration (None: the schedule's)."""
        raise NotImplementedError(f"{cls.kind} is not an event signal")

    def render_events(self, events: list[Event], n0: int, n: int, rate: int) -> np.ndarray:
        """The given events (each whole, from its start) in samples [n0, n0 + n) at `rate`; 0 elsewhere."""
        raise NotImplementedError(f"{self.kind} is not an event signal")

    def event_bounds(self, ev: Event) -> tuple[float, float]:
        """Event signals: the time span outside which the event's samples are exactly 0 (filter tails included)."""
        return ev.start, ev.end

    # --- wideband signals (M7) ---

    def render_wide(
        self, events: list[Event], h0: int, nhops: int, hop: int, fs: int, n0_w: float, f_offset_hz: float = 0.0
    ) -> np.ndarray:
        """Wideband signals: hops h0 .. h0 + nhops - 1 at fs (complex64, sqrt(mW)) of the given events; n0_w is the
        noise density k T_sys in mW/Hz that relative levels refer to, f_offset_hz the emitter's frequency minus fc."""
        raise NotImplementedError(f"{self.kind} is not a wideband signal")


def signal_class(kind: str) -> type[Signal]:
    k = get_kind(kind)
    if kind not in IMPLEMENTED:
        raise NotImplementedError(f"signal kind {kind!r} is rendered from milestone {k.milestone}")
    from assist.signals import am, cw, d8psk, gfsk, lora, noise, oqpsk, oqpsk_burst, sdpsk

    classes: dict[str, type[Signal]] = {
        "cw": cw.CW,
        "oqpsk": oqpsk.OQPSK,
        "oqpsk_burst": oqpsk_burst.OQPSKBurst,
        "sdpsk": sdpsk.SDPSK,
        "lora": lora.LoRa,
        "d8psk": d8psk.D8PSK,
        "gfsk": gfsk.GFSK,
        "am": am.AM,
        "noise_band": noise.NoiseBand,
        "noise_comb": noise.NoiseComb,
    }
    return classes[kind]


def check_params(kind: str, params: dict[str, Any]) -> dict[str, Any]:
    """The kind's parameters validated, with defaults filled in (kinds not implemented yet: unchanged)."""
    if kind not in IMPLEMENTED:
        return params
    return signal_class(kind).Params.model_validate(params).model_dump(mode="json")


def make_signal(kind: str, params: dict[str, Any], keys: Keys, dsp: ModuleType) -> Signal:
    cls = signal_class(kind)
    return cls(cls.Params.model_validate(params), keys, dsp)


def event_maker(kind: str, params: dict[str, Any]) -> tuple[type[Signal], SignalParams]:
    """The class and validated parameters of an event (or wideband) signal (for its schedule's event maker)."""
    cls = signal_class(kind)
    if cls.mode == "continuous":
        raise ValueError(f"{kind} is not an event signal")
    return cls, cls.Params.model_validate(params)


def int_range(v: int | tuple[int, int]) -> tuple[int, int]:
    """A per-event parameter given as a number or an inclusive [lo, hi] range."""
    return (v, v) if isinstance(v, int) else v


def float_range(v: float | tuple[float, float]) -> tuple[float, float]:
    return (float(v), float(v)) if isinstance(v, int | float) else (float(v[0]), float(v[1]))


def lcm(a: int, b: int) -> int:
    return a * b // math.gcd(a, b)
