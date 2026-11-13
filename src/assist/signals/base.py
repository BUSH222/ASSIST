"""Signal kinds (plan §3.4, §5).

A kind is a `Signal` subclass registered under a name with `@register_signal("name")`; the core kinds live in this
package, external ones in modules a scenario lists under `plugins:` (PLAN.md). Parameters are a pydantic model per
kind (unknown keys are errors); validation fills in the defaults, so a resolved scenario lists every parameter.

Three modes:
- `continuous`: `render(n0, n, rate)` gives samples at any absolute narrow index (cw, oqpsk, oqpsk_burst, sdpsk,
  noise_band);
- `event`: the emitter's schedule places events; `draw_event` draws an event's parameters and duration (from the
  schedule's keyed slot stream; a duration of None means the schedule's duration distribution decides) and
  `render_events` renders the events overlapping a window (lora, d8psk, gfsk, am);
- `wideband`: interference inside the receiver, rendered at the full sample rate in the time domain and added
  before the ADC, bypassing the synthesiser and the receiver response (noise_comb, the USB humps).

Random content (payload bits, frames, burst symbols) comes from `Keys`, the emitter's `RngScope`: 64-bit keys per
(source, emitter, purpose) for the counter-based `random_bytes` kernel, whose counter is an absolute index (frame,
symbol chunk, segment, event start), so any window is rendered on its own (plan §3.2); `Keys.stream` gives a numpy
Generator under the same prefix for per-event draws. A signal never holds the seed or the `KeyedRng` itself.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from assist import registry
from assist.rng import RngScope

if TYPE_CHECKING:
    from types import ModuleType

    import numpy as np

    from assist.emitters.schedules import Event
    from assist.rng import KeyedRng


@dataclass(frozen=True)
class SignalKind:
    name: str
    mode: Literal["continuous", "event", "wideband"]


class _Register(Protocol):
    def __call__[C: type[Signal]](self, cls: C, /) -> C: ...


def register_signal(name: str) -> _Register:
    """Class decorator: make a `Signal` subclass available as signal kind `name`.

    Core kinds and external ones use the same decorator. An external kind can be used by a scenario that lists its
    module under `plugins:` (see `assist.plugins`)."""
    reg = registry.signals.register(name)

    def deco[C: type[Signal]](cls: C, /) -> C:
        if not (isinstance(cls, type) and issubclass(cls, Signal)):
            raise TypeError(f"register_signal({name!r}) needs a Signal subclass, got {cls!r}")
        if not isinstance(getattr(cls, "Params", None), type) or not issubclass(cls.Params, SignalParams):
            raise TypeError(f"signal {name!r}: {cls.__name__}.Params must be a SignalParams subclass")
        if cls.mode not in ("continuous", "event", "wideband"):
            raise TypeError(f"signal {name!r}: mode must be continuous, event or wideband, got {cls.mode!r}")
        setattr(cls, "kind", name)  # noqa: B010 (a ClassVar of the subclass, set through the type variable)
        reg(cls)
        return cls

    return deco


def signal_class(kind: str) -> type[Signal]:
    try:
        cls = registry.signals.get(kind)
    except registry.RegistryError as e:
        raise ValueError(str(e)) from None
    assert issubclass(cls, Signal)
    return cls


def get_kind(name: str) -> SignalKind:
    return SignalKind(name, signal_class(name).mode)


class SignalParams(BaseModel):
    """Base of every kind's parameter model: unknown keys are errors."""

    model_config = ConfigDict(extra="forbid")


class Keys(RngScope):
    """Random streams of one emitter: the scope (source_id, emitter_id), so every key is
    (source_id, emitter_id, purpose, ...) from the scenario seed. `key64` feeds the counter-based kernels, `stream`
    gives a numpy Generator (see `RngScope`)."""

    __slots__ = ("emitter_id", "source_id")

    def __init__(self, rng: KeyedRng, source_id: str, emitter_id: str, namespace: tuple[str | int, ...] = ()) -> None:
        super().__init__(rng, (*namespace, source_id, emitter_id))
        self.source_id, self.emitter_id = source_id, emitter_id

    def namespaced(self, *namespace: str | int) -> Keys:
        """The same emitter's keys under a namespace: (*namespace, source_id, emitter_id, ...)."""
        return Keys(self._rng, self.source_id, self.emitter_id, namespace)


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

    # --- wideband signals ---

    def render_wide(
        self, events: list[Event], h0: int, nhops: int, hop: int, fs: int, n0_w: float, f_offset_hz: float = 0.0
    ) -> np.ndarray:
        """Wideband signals: hops h0 .. h0 + nhops - 1 at fs (complex64, sqrt(mW)) of the given events; n0_w is the
        noise density k T_sys in mW/Hz that relative levels refer to, f_offset_hz the emitter's frequency minus fc."""
        raise NotImplementedError(f"{self.kind} is not a wideband signal")


def check_params(kind: str, params: dict[str, Any]) -> dict[str, Any]:
    """The kind's parameters validated, with defaults filled in."""
    return signal_class(kind).Params.model_validate(params).model_dump(mode="json")


def make_signal(kind: str, params: dict[str, Any], keys: Keys, dsp: ModuleType) -> Signal:
    """The signal of one emitter. `keys` is the emitter's scope (source_id, emitter_id); an external kind gets it
    under ("signal", kind), so its keys cannot meet a core kind's or another module's."""
    cls = signal_class(kind)
    if registry.signals.is_external(kind):
        keys = keys.namespaced("signal", kind)
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
