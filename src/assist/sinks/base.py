"""Sink interface (PLAN.md R3): where the station's uint8 IQ stream goes.

A sink is a class registered under a name with `@register_sink("name")`; a scenario lists the sinks it uses, with
their settings, under `sinks:` (a list of one-key mappings), and `assist run` drives them:

    sinks:
      - wav: {rf64: true}
      - fft: {width: 16384, row_seconds: 1}

Sinks do not launch themselves. The runner (`sinks/runner.py`) builds the Renderer once, writes `run.json`, maps
errors to exit codes, and hands the stream to the sinks. There are two flavours:

- **push** sinks (`Sink`: wav, cu8, fft) are given the stream piece by piece (`write`), in order, as fast as it
  renders; several can share one run.
- **serving** sinks (`ServingSink`: rtl_tcp) own an event loop: they are given a `RenderAhead` (one render thread
  filling a ring) and pull from it at their own pace. A run has exactly one of them and nothing else.

Settings come from the scenario only (`--set 'sinks[wav].rf64=true'` overrides one), validated by the sink's
`Params` model; there are no sink-specific command-line options.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, override

from pydantic import BaseModel, ConfigDict, Field, field_validator

from assist import registry
from assist.timebase import parse_duration

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from assist.rng import RngScope
    from assist.sinks.rtltcp import RenderAhead
    from assist.sinks.wavfile import OutFile


class SinkParams(BaseModel):
    """Base of every sink's parameter model: unknown keys are errors."""

    model_config = ConfigDict(extra="forbid")


class ServingParams(SinkParams):
    """What every serving sink has: where it listens and how the stream is timed."""

    host: str = Field("127.0.0.1", description="Listen address.")
    port: int = Field(1234, ge=0, le=65535, description="TCP port (0: any free port).")
    pacing: Literal["realtime", "unpaced"] = Field(
        "realtime", description="realtime, or unpaced (as fast as the clients read)."
    )
    lookahead: float = Field(1.0, gt=0, description="Render-ahead ring in seconds (strings like 4s are accepted).")

    @field_validator("lookahead", mode="before")
    @classmethod
    def _lookahead(cls, v: Any) -> float:
        return parse_duration(v)


@dataclass(frozen=True)
class SinkContext:
    """What a sink is built from: the stream it will be given, and where its outputs go."""

    fs: int
    fc_hz: float
    n0: int  # absolute index (Unix time * fs) of the stream's first sample
    n: int | None  # samples in the stream (None: endless, serving sinks only)
    out: Path  # output directory of the run
    tz: str  # the station's time zone (file names use local time)
    scope: RngScope  # this sink's randomness: keys under ("sink", sink name)
    scenario_hash: str
    wall_start: float | None = None  # serving, live: the wall time (Unix s) of the first sample


class Sink(ABC):
    """A push sink. Lifecycle: constructed (validate, plan), `open`, `write` ... , then `close` or `abort`."""

    kind: ClassVar[str]
    Params: ClassVar[type[SinkParams]] = SinkParams
    serving: ClassVar[bool] = False

    def __init__(self, params: SinkParams, ctx: SinkContext) -> None:
        self.p, self.ctx = params, ctx

    def planned(self) -> int:
        """How many samples this sink takes, counted from the start of the stream (at most the stream's length).
        A sink that needs fewer finishes early; the run ends when every sink has what it planned."""
        assert self.ctx.n is not None
        return self.ctx.n

    def files(self) -> list[OutFile]:
        """The files this sink writes (checked for overwriting before the run, listed in run.json after it)."""
        return []

    def open(self) -> None:  # noqa: B027 (optional hook)
        """Called once before the first `write`."""

    @abstractmethod
    def write(self, u8: np.ndarray) -> None:
        """The next piece of the stream: interleaved uint8 I, Q, any length."""

    def close(self) -> None:  # noqa: B027 (optional hook)
        """The stream is complete: finish the outputs. Raise if the sink did not get what it planned."""

    def abort(self) -> None:  # noqa: B027 (optional hook)
        """The run stopped early (error or Ctrl-C): keep whatever is usable."""

    def info(self) -> dict[str, Any]:
        """Facts about what was written, for run.json."""
        return {}

    def summary(self) -> str | None:
        """One line for the end of the run (None: nothing to add)."""
        return None


class ServingSink(Sink):
    """A sink that serves the stream to network clients. Its `Params` derive from `ServingParams`."""

    Params: ClassVar[type[SinkParams]] = ServingParams
    serving: ClassVar[bool] = True

    @override
    def write(self, u8: np.ndarray) -> None:
        raise NotImplementedError(f"{self.kind} is a serving sink")

    @property
    def sp(self) -> ServingParams:
        p = self.p
        assert isinstance(p, ServingParams)
        return p

    @abstractmethod
    async def serve(self, ahead: RenderAhead) -> None:
        """Start `ahead`, serve until the stream ends, then close every connection. A render error is raised."""


class _Register(Protocol):
    def __call__[C: type[Sink]](self, cls: C, /) -> C: ...


def register_sink(name: str) -> _Register:
    """Class decorator: make a `Sink` subclass available as the sink `name` (core and external alike)."""
    reg = registry.sinks.register(name)

    def deco[C: type[Sink]](cls: C, /) -> C:
        if not (isinstance(cls, type) and issubclass(cls, Sink)):
            raise TypeError(f"register_sink({name!r}) needs a Sink subclass, got {cls!r}")
        if not isinstance(getattr(cls, "Params", None), type) or not issubclass(cls.Params, SinkParams):
            raise TypeError(f"sink {name!r}: {cls.__name__}.Params must be a SinkParams subclass")
        if cls.serving and not (issubclass(cls, ServingSink) and issubclass(cls.Params, ServingParams)):
            raise TypeError(f"sink {name!r}: a serving sink derives from ServingSink, its Params from ServingParams")
        setattr(cls, "kind", name)  # noqa: B010 (a ClassVar of the subclass, set through the type variable)
        reg(cls)
        return cls

    return deco


def sink_class(name: str) -> type[Sink]:
    try:
        cls = registry.sinks.get(name)
    except registry.RegistryError as e:
        raise ValueError(str(e)) from None
    assert issubclass(cls, Sink)
    return cls


def check_sinks(entries: list[Any]) -> list[dict[str, Any]]:
    """A scenario's `sinks:` validated: one-key mappings {sink name: settings}, each sink at most once; the settings
    are checked by the sink's `Params`, and their defaults filled in."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for e in entries:
        if isinstance(e, str):
            e = {e: None}
        if not isinstance(e, dict) or len(e) != 1:
            raise ValueError(f"a sinks entry is a one-key mapping (sink name: settings), got {e!r}")
        ((name, params),) = e.items()
        cls = sink_class(str(name))
        if name in seen:
            raise ValueError(f"sink {name!r} is listed twice")
        seen.add(name)
        if params is not None and not isinstance(params, dict):
            raise ValueError(f"sink {name!r}: settings must be a mapping, got {params!r}")
        out.append({name: cls.Params.model_validate(params or {}).model_dump(mode="json")})
    return out
