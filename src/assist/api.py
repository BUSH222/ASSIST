"""What an external module imports (PLAN.md). Everything a plugin needs comes from here; other `assist` modules are
internal and may change.

    from assist import api

    ASSIST_API = "1.0"   # the API_VERSION this module was written for (required)

    class MyParams(api.SignalParams):
        rb: int = 9600

    @api.register_signal("my_kind")
    class MySignal(api.Signal):
        Params = MyParams
        ...

A signal gets its parameters (`self.p`), its random keys (`self.keys`, an `RngScope` of this emitter) and the kernels
(`self.dsp`). Rules a signal must keep:

- **Absolute indices.** `render(n0, n, rate)` and `render_events(events, n0, n, rate)` return the samples of absolute
  narrow indices [n0, n0 + n) (sample n is at Unix time n / rate). Rendering a span whole or in pieces must give the
  same samples.
- **Keyed randomness only.** Random content comes from `self.keys`: `key64(purpose)` with a counter-based kernel
  (`dsp.random_bytes(key, block, n)`, `dsp.white(...)`) whose counter is an absolute index, or `stream(purpose,
  index)` for a few draws about one event. No `np.random.default_rng()`, no global state.
- **Unit mean power while on.** Level, Doppler, schedule and fading are applied by the engine.

A platform (`Platform`, `register_platform`) says where a source is: it is built from a `PlatformContext` (its
validated `platform_params`, the station, an `RngScope`) and gives `geometry(t)` and `passes(t0, t1)` as pure
functions of absolute time, plus `max_speed_km_s()`; or it sets `fixed = True` and has no geometry at all. Its class
attributes declare what a source on it may have (see `Platform`). An external platform's emitters take a received
level (`cn0_dbhz` / `rx_dbm`); orbits, the catalog and the link budget belong to the core `satellite`.

A sink (`Sink`, `register_sink`) is where the uint8 IQ stream goes. `assist run` drives it: a push sink is built
from its validated settings and a `SinkContext`, says how many samples it takes (`planned`) and which files it writes
(`files`), then gets `open`, `write(u8)` for every piece, and `close` (or `abort`). A serving sink derives from
`StreamSink` and supplies a `StreamServer` subclass: rendering ahead, pacing, fan-out and the slow-client guard are
shared, the subclass only says what goes on the wire (`greeting`, `read_client`, `encode`).

No C++ is involved in writing a signal: it is numpy code at the narrow rate. The compiled kernels do the wideband
work afterwards (synthesis to the full sample rate, noise, ADC) for every kind alike.

**Kernels.** Of `self.dsp`, two functions belong to this API, because they are how a signal gets random content that
is addressed by an absolute counter:

- `dsp.random_bytes(key, block, n)`: n random bytes (uint8) of block `block`;
- `dsp.white(key, h0, nhops, hop)`: complex white Gaussian noise (unit variance per component) of hops
  h0 .. h0 + nhops - 1, `hop` samples each.

The other kernels (`shape`, `chirp`, `nco`, `synth`, ...) serve the core kinds and the engine and may change without
an API version change; `ShapedSignal` is the supported way to pulse shaping.

**Versions.** `API_VERSION` is "major.minor" and covers the names in `__all__` below, the rules above and the two
kernels. Additions raise the minor; a change that can break a plugin raises the major. A plugin declares the version
it was written for (`ASSIST_API = "1.0"`): it loads when the major is the same and its minor is not above this one.

`assist.testing` has conformance checks for the three kinds of module (`check_signal`, `check_platform`,
`check_sink`), for a plugin's own tests.
"""

from assist.emitters.schedules import Event
from assist.orbits.geometry import Geometry, Pass
from assist.platforms.base import Platform, PlatformContext, PlatformParams, register_platform
from assist.plugins import API_VERSION
from assist.rng import RngScope
from assist.signals.base import (
    Keys,
    Signal,
    SignalParams,
    float_range,
    int_range,
    register_signal,
)
from assist.signals.pulse import ShapedParams, ShapedSignal, rrc_taps, unpack
from assist.sinks.base import ServingParams, ServingSink, Sink, SinkContext, SinkParams, register_sink
from assist.sinks.stream import RenderAhead, StreamServer, StreamSink
from assist.sinks.wavfile import OutFile

__all__ = [
    "API_VERSION",
    "Event",
    "Geometry",
    "Keys",
    "OutFile",
    "Pass",
    "Platform",
    "PlatformContext",
    "PlatformParams",
    "RenderAhead",
    "RngScope",
    "ServingParams",
    "ServingSink",
    "ShapedParams",
    "ShapedSignal",
    "Signal",
    "SignalParams",
    "Sink",
    "SinkContext",
    "SinkParams",
    "StreamServer",
    "StreamSink",
    "float_range",
    "int_range",
    "register_platform",
    "register_signal",
    "register_sink",
    "rrc_taps",
    "unpack",
]
