"""Conformance checks for external modules: does a signal, a platform or a sink keep the rules of `assist.api`?

For a plugin's own tests:

    from assist import testing

    mod = testing.load_plugin("examples/plugins/pretty_gfsk.py")
    testing.check_signal(mod.PrettyGfsk, {"burst_ms": [150, 300]})

Each check raises `ConformanceError` (an AssertionError) that says what is wrong, and returns nothing when the
module conforms. A class may be given as the class or as its registered name (after `load_plugin`).

- `check_signal`: parameters are validated (an unknown key is refused); the narrow rate is consistent (the signal
  is rendered at the rate the engine would pick for it); rendering a span whole and in pieces gives the same samples
  (fresh instances per piece, so nothing may be carried over); the output is complex64 with unit mean power while
  on, and an event signal is exactly 0 outside its events' bounds.
- `check_platform`: unknown parameters are refused; the same seed gives the same platform; `geometry` is a pure
  function of time (whole vs pieces, and against `geometry_ms`); `passes` do not depend on the window asked for;
  `max_speed_km_s` bounds the range rate.
- `check_sink`: unknown settings are refused; a push sink writes the files it plans, and the same bytes however the
  stream is cut into pieces; a serving sink serves a short stream to a client and stops at its end.

The checks use only the public interface, with a seed of their own: they do not need a scenario.
"""

import asyncio
import itertools
import json
import socket
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import ValidationError

from assist import plugins, registry
from assist.emitters.schedules import Event
from assist.engine.control import RenderError, narrow_rate
from assist.platforms.base import Platform, PlatformContext
from assist.rng import KeyedRng
from assist.signals.base import Keys, Signal
from assist.sinks.base import ServingParams, ServingSink, Sink, SinkContext
from assist.sinks.stream import RenderAhead
from assist.station import FALLBACK_LAT_DEG, FALLBACK_LON_DEG, Station

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from types import ModuleType

__all__ = ["ConformanceError", "check_platform", "check_signal", "check_sink", "load_plugin"]

T0 = 1_793_750_400.0  # 2026-11-04T00:00:00Z: where the checks put their windows unless told otherwise
FS = 2_400_000
FC_HZ = 137.5e6
UNKNOWN = "no_such_parameter_"  # a key no Params model has
WIDE_HOP = 512  # samples per hop in the wideband check


class ConformanceError(AssertionError):
    pass


def _fail(what: str, message: str) -> ConformanceError:
    return ConformanceError(f"{what}: {message}")


def load_plugin(entry: str | dict[str, Any], *dirs: Path) -> ModuleType:
    """Load one plugin (a path, a name, or a `plugins:` entry) and make it the active one, as a scenario listing it
    would. Relative paths are looked up in `dirs` (default: the working directory). Returns its module."""
    e = plugins.from_cli(entry) if isinstance(entry, str) else entry
    plugins.activate([e], dirs or (Path.cwd(),))
    return plugins.active()[0].module


def _class[T](category: str, given: type[T] | str, base: type[T]) -> type[T]:
    found: Any = given
    if isinstance(given, str):
        try:
            found = registry.ALL[category].get(given)
        except registry.RegistryError as e:
            raise ConformanceError(str(e)) from None
    if not (isinstance(found, type) and issubclass(found, base)):
        raise ConformanceError(f"{found!r} is not a {base.__name__} subclass")
    if not isinstance(getattr(found, "kind", None), str):
        raise ConformanceError(f"{found.__name__} is not registered (no register_{category} decorator)")
    return found


def _params(what: str, cls: Any, params: dict[str, Any] | None) -> Any:
    """The validated parameters; an unknown key must be refused."""
    given = dict(params or {})
    try:
        p = cls.Params.model_validate(given)
    except ValidationError as e:
        raise _fail(what, f"the given parameters are not valid: {e}") from None
    try:
        cls.Params.model_validate({**given, UNKNOWN: 1})
    except ValidationError:
        return p
    raise _fail(what, "an unknown parameter is accepted (derive Params from the api base class: extra='forbid')")


def _cuts(n: int, pieces: int, g: np.random.Generator) -> list[int]:
    """0, n and pieces - 1 distinct interior points, odd ones on purpose."""
    inner = sorted({int(x) | 1 for x in g.integers(1, max(2, n - 1), pieces - 1) if 0 < (int(x) | 1) < n})
    return [0, *inner, n]


# --- signals ---


def check_signal(
    signal: type[Signal] | str,
    params: dict[str, Any] | None = None,
    *,
    seed: int = 1,
    t0: float = T0,
    seconds: float = 2.0,
    events: Sequence[Event] | None = None,
    event_s: float = 0.25,
    power_tol_db: float = 0.5,
    pieces: int = 5,
    atol: float = 0.0,
) -> None:
    """Check a signal kind (see the module text).

    `seconds`: the span rendered. Event and wideband signals get `events`, or events drawn with the kind's own
    `draw_event` laid out over the span (an event whose `draw_event` leaves the duration to the schedule lasts
    `event_s`). `power_tol_db`: how far the mean power while on may be from 1 (not checked for wideband signals,
    whose level is absolute; a continuous signal that is bursty by design needs a wider tolerance). `atol`: the
    largest difference allowed between a span rendered whole and in pieces; the default 0 asks for identical samples,
    which is what makes a render independent of how the engine cuts it."""
    from assist.backend import dsp

    cls = _class("signal", signal, Signal)
    what = f"signal {cls.kind}"
    p = _params(what, cls, params)

    def make() -> Signal:
        return cls(p, Keys(KeyedRng(seed), "source", "emitter", ("signal", cls.kind)), dsp)

    sig = make()
    rate, step, bw = sig.preferred_rate(), sig.rate_step(), sig.bandwidth_hz()
    if not isinstance(rate, int) or not isinstance(step, int) or rate <= 0 or step <= 0:
        raise _fail(what, f"preferred_rate() and rate_step() must be positive integers, got {rate!r} and {step!r}")
    if rate % step:
        raise _fail(what, f"preferred_rate() {rate} is not a multiple of rate_step() {step}")
    if not bw >= 0:
        raise _fail(what, f"bandwidth_hz() must be >= 0, got {bw!r}")
    try:  # the narrow rate the engine would pick at 2.4 MS/s for an emitter at rest
        rate = narrow_rate(bw, 0.0, rate, FS, FS // 50, step)
    except RenderError as e:
        raise _fail(what, str(e)) from None
    g = KeyedRng(seed).stream("testing", "cuts")
    if cls.mode == "wideband":
        _check_wideband(what, cls, p, make, seed, t0, seconds, events, event_s, pieces, g)
        return
    n0 = round(t0 * rate) + 7
    n = round(seconds * rate)
    cuts = _cuts(n, pieces, g)
    if cls.mode == "continuous":
        whole = _samples(what, sig.render(n0, n, rate), n)
        parts = [_samples(what, make().render(n0 + a, b - a, rate), b - a) for a, b in itertools.pairwise(cuts)]
        on = np.ones(n, dtype=bool)
    else:
        evs = list(events) if events is not None else _draw_events(what, cls, p, seed, t0, seconds, event_s)
        if not evs:
            raise _fail(what, "no events to render (give `events`, or a longer span)")
        bounds = [sig.event_bounds(ev) for ev in evs]

        def render(s: Signal, a: int, k: int) -> np.ndarray:
            lo, hi = a / rate, (a + k) / rate
            touching = [ev for ev, (b0, b1) in zip(evs, bounds, strict=True) if b1 > lo and b0 < hi]
            return _samples(what, s.render_events(touching, a, k, rate), k)

        whole = render(sig, n0, n)
        parts = [render(make(), n0 + a, b - a) for a, b in itertools.pairwise(cuts)]
        t = (n0 + np.arange(n, dtype=np.int64)) / rate
        on = np.zeros(n, dtype=bool)
        for b0, b1 in bounds:
            on |= (t >= b0) & (t < b1)
        outside = np.abs(whole[~_grow(on, 2)])
        if outside.size and outside.max() > 0:
            raise _fail(what, "samples outside the events' bounds are not 0 (override event_bounds for filter tails)")
    diff = np.abs(np.concatenate(parts) - whole)
    if diff.max() > atol:
        bad = int(np.flatnonzero(diff > atol)[0])
        raise _fail(
            what,
            f"rendering a span in pieces differs from rendering it whole (by up to {diff.max():.3g}, first at sample "
            f"{bad} of {n}; pieces cut at {cuts[1:-1]}): samples must depend on absolute indices only",
        )
    if not on.any():
        raise _fail(what, "the signal is never on in the span")
    power = float(np.mean(np.abs(whole[on].astype(np.complex128)) ** 2))
    db = 10 * np.log10(power) if power > 0 else -np.inf
    if not abs(db) <= power_tol_db:
        raise _fail(what, f"mean power while on is {db:+.2f} dB, not 0 dB within {power_tol_db} dB (unit power)")


def _samples(what: str, x: Any, n: int) -> np.ndarray:
    if not isinstance(x, np.ndarray) or x.dtype != np.complex64 or x.shape != (n,):
        got = f"{x.dtype} {x.shape}" if isinstance(x, np.ndarray) else type(x).__name__
        raise _fail(what, f"a render must return complex64 of shape ({n},), got {got}")
    if not np.isfinite(x.view(np.float32)).all():
        raise _fail(what, "a render has non-finite samples")
    return x


def _grow(mask: np.ndarray, k: int) -> np.ndarray:
    """The mask widened by k samples each way (rounding of bounds to samples)."""
    out = mask.copy()
    for s in range(1, k + 1):
        out[s:] |= mask[:-s]
        out[:-s] |= mask[s:]
    return out


def _draw_events(what: str, cls: Any, p: Any, seed: int, t0: float, seconds: float, event_s: float) -> list[Event]:
    """Events from the kind's own `draw_event`, one after another with gaps, inside [t0, t0 + seconds)."""
    scope = KeyedRng(seed).scope("testing", "events")
    upper = cls.max_duration_s(p)
    out: list[Event] = []
    t = t0 + 0.0123
    for i in itertools.count():
        d, ev_params = cls.draw_event(p, scope.stream(i))
        if d is None:
            d = event_s if upper is None else min(event_s, upper)
        if not d > 0:
            raise _fail(what, f"draw_event gave a duration of {d!r}")
        if upper is not None and d > upper + 1e-9:
            raise _fail(what, f"draw_event gave {d} s, above max_duration_s() = {upper} s")
        if t + d > t0 + seconds - 0.01 and out:
            return out
        if i >= 1000 or t + d > t0 + seconds - 0.01:
            raise _fail(what, f"an event of {d:.3f} s does not fit in the span of {seconds} s (give a longer one)")
        out.append(Event(t, t + d, dict(ev_params)))
        t += d + 0.0371
    return out


def _check_wideband(
    what: str,
    cls: Any,
    p: Any,
    make: Any,
    seed: int,
    t0: float,
    seconds: float,
    events: Sequence[Event] | None,
    event_s: float,
    pieces: int,
    g: np.random.Generator,
) -> None:
    evs = list(events) if events is not None else _draw_events(what, cls, p, seed, t0, seconds, event_s)
    h0 = round(t0 * FS / WIDE_HOP)
    nh = round(seconds * FS / WIDE_HOP)
    n0_w = 4e-18  # k T for about 290 K, in mW/Hz

    def render(s: Signal, a: int, k: int) -> np.ndarray:
        return _samples(what, s.render_wide(evs, a, k, WIDE_HOP, FS, n0_w, 0.0), k * WIDE_HOP)

    whole = render(make(), h0, nh)
    cuts = _cuts(nh, pieces, g)
    parts = [render(make(), h0 + a, b - a) for a, b in itertools.pairwise(cuts)]
    if not np.array_equal(np.concatenate(parts), whole):
        raise _fail(what, "rendering hops in pieces differs from rendering them whole")
    if not np.abs(whole).max() > 0:
        raise _fail(what, "the signal is never on in the span")


# --- platforms ---


def check_platform(
    platform: type[Platform] | str,
    params: dict[str, Any] | None = None,
    *,
    seed: int = 1,
    t0: float = T0 + 43_200.0,
    seconds: float = 120.0,
    station: Station | None = None,
    horizon_deg: float = 0.0,
    source: Any = None,
) -> None:
    """Check a platform (see the module text). `source`: the resolved source handed to the platform (None suits an
    external platform, which reads its `platform_params` only). A fixed platform has no geometry to check."""
    cls = _class("platform", platform, Platform)
    what = f"platform {cls.kind}"
    p = _params(what, cls, params)
    st = station or Station(FALLBACK_LAT_DEG, FALLBACK_LON_DEG)

    def build(sid: str = "source", s: int = seed) -> Platform:
        scope = KeyedRng(s).scope("platform", cls.kind, sid)
        return cls.build(PlatformContext(sid, sid, p, st, horizon_deg, scope, source))

    plat = build()
    if not isinstance(plat, Platform):
        raise _fail(what, f"build() must return a Platform, got {type(plat).__name__}")
    if cls.fixed:
        return
    k = round(t0 * 1000) + np.arange(0, round(seconds * 1000), 50, dtype=np.int64)
    t = k / 1000.0
    fields = ("el_deg", "az_deg", "range_m", "range_rate_m_s", "off_nadir_deg")

    def geo(pl: Platform, tt: np.ndarray, ms: bool = False) -> dict[str, np.ndarray]:
        gm = pl.geometry_ms(tt) if ms else pl.geometry(tt)
        out = {f: np.asarray(getattr(gm, f), dtype=np.float64) for f in fields}
        for f, v in out.items():
            if v.shape != tt.shape:
                raise _fail(what, f"geometry: {f} has shape {v.shape} for {tt.shape[0]} times")
        return out

    whole = geo(plat, t)
    for f in ("el_deg", "az_deg", "range_m", "range_rate_m_s"):
        if not np.isfinite(whole[f]).all():
            raise _fail(what, f"geometry: {f} is not finite")
    if (whole["range_m"] <= 0).any():
        raise _fail(what, "geometry: range_m must be positive")
    again = geo(build(), t)
    half = len(t) // 2
    split = [geo(build(), t[:half]), geo(build(), t[half:])]
    ms = geo(plat, k, ms=True)
    for f in fields:
        if not np.array_equal(again[f], whole[f], equal_nan=True):
            raise _fail(what, f"two platforms built from the same seed and source differ in {f}")
        if not np.array_equal(np.concatenate([s[f] for s in split]), whole[f], equal_nan=True):
            raise _fail(what, f"geometry: {f} depends on which times are asked together (not a pure function of time)")
    if not np.allclose(ms["range_m"], whole["range_m"], rtol=0, atol=1e-2):
        raise _fail(what, "geometry_ms and geometry disagree by more than 1 cm in range")
    vmax = plat.max_speed_km_s()
    seen = float(np.abs(whole["range_rate_m_s"]).max()) / 1000.0
    if not vmax >= seen * (1 - 1e-9):
        raise _fail(what, f"max_speed_km_s() = {vmax} is below the range rate seen ({seen:.6f} km/s)")
    # Range rate against the derivative of the range (on the 1 ms grid, where the engine takes it)
    kk = round(t0 * 1000) + np.arange(2001, dtype=np.int64)
    fine = geo(plat, kk, ms=True)
    slope = (fine["range_m"][2:] - fine["range_m"][:-2]) / 2e-3
    tol = 1e-3 * max(1.0, float(np.abs(fine["range_rate_m_s"]).max())) + 0.05
    if not np.allclose(slope, fine["range_rate_m_s"][1:-1], rtol=0, atol=tol):
        raise _fail(what, "range_rate_m_s is not the time derivative of range_m")
    # Passes: whole, and the same whichever window asks
    a, b = t0, t0 + seconds
    wide = plat.passes(a - 3600.0, b + 3600.0)
    known = {(q.aos, q.los, q.aos_key) for q in wide}
    for q in wide:
        if not q.los > q.aos:
            raise _fail(what, f"a pass ends before it starts ({q.aos} .. {q.los})")
        if q.aos_key is None:
            raise _fail(what, "a pass has no aos_key (random draws per pass are keyed by it)")
    if len({q.aos_key for q in wide}) != len(wide):
        raise _fail(what, "two passes share an aos_key")
    for lo, hi in ((a, b), (a, a + seconds / 3), (a + seconds / 2, b)):
        for q in build().passes(lo, hi):
            if not (q.los > lo and q.aos < hi):
                raise _fail(what, f"passes({lo}, {hi}) returned a pass outside the window ({q.aos} .. {q.los})")
            if (q.aos, q.los, q.aos_key) not in known:
                raise _fail(what, "a pass depends on the window asked for (its aos, los or aos_key changed)")
        want = sum(1 for q in wide if q.los > lo and q.aos < hi)
        if len(plat.passes(lo, hi)) != want:
            raise _fail(what, f"passes({lo}, {hi}) misses passes that a wider window gives")
    keys = plat.pass_keys(t, plat.passes(a, b))
    if np.asarray(keys).shape != t.shape:
        raise _fail(what, "pass_keys must give one key per time")


# --- sinks ---


def _stream(n0: int, n: int, seed: int) -> np.ndarray:
    """A deterministic uint8 I, Q stream of n samples: noise around mid-scale plus a tone."""
    g = KeyedRng(seed).stream("testing", "stream")
    j = n0 + np.arange(n, dtype=np.int64)
    tone = 40.0 * np.exp(2j * np.pi * ((j % FS) * (100_000 / FS)))
    x = tone + 12.0 * (g.standard_normal(n) + 1j * g.standard_normal(n))
    out = np.empty(2 * n, dtype=np.uint8)
    out[0::2] = np.clip(np.rint(127.5 + x.real), 0, 255)
    out[1::2] = np.clip(np.rint(127.5 + x.imag), 0, 255)
    return out


class _Source:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def iter_u8(self, n0: int, n: int) -> Iterator[np.ndarray]:
        yield _stream(n0, n, self.seed)[: 2 * n]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def check_sink(
    sink: type[Sink] | str,
    params: dict[str, Any] | None = None,
    *,
    seed: int = 1,
    t0: float = T0,
    seconds: float = 1.0,
    fs: int = FS,
    out: Path | None = None,
) -> None:
    """Check a sink (see the module text). `out`: a directory for what it writes (default: a temporary one, removed
    afterwards). A serving sink is run on a free local port with `pacing: unpaced`, whatever `params` say."""
    cls = _class("sink", sink, Sink)
    what = f"sink {cls.kind}"
    _params(what, cls, params)
    n0 = round(t0 * fs)
    n = round(seconds * fs)
    with tempfile.TemporaryDirectory() as tmp:
        base = out or Path(tmp)
        if cls.serving:
            _check_serving(what, cls, params, seed, n0, n, fs, base)
        else:
            _check_push(what, cls, params, seed, n0, n, fs, base)


def _context(cls: Any, seed: int, n0: int, n: int | None, fs: int, out: Path) -> SinkContext:
    out.mkdir(parents=True, exist_ok=True)
    return SinkContext(fs, FC_HZ, n0, n, out, "+00:00", KeyedRng(seed).scope("sink", cls.kind), "0" * 64)


def _check_push(
    what: str, cls: Any, params: dict[str, Any] | None, seed: int, n0: int, n: int, fs: int, base: Path
) -> None:
    data = _stream(n0, n, seed)
    results: list[dict[str, bytes]] = []
    for run, piece in enumerate((n, 16_384, 99_991)):
        out = base / f"run{run}"
        sink = cls(cls.Params.model_validate(params or {}), _context(cls, seed, n0, n, fs, out))
        take = sink.planned()
        if not isinstance(take, int) or not 0 < take <= n:
            raise _fail(what, f"planned() must be in 1 .. {n} (the stream's length), got {take!r}")
        files = sink.files()
        paths = [f.path for f in files]
        if len(set(paths)) != len(paths):
            raise _fail(what, "files() lists a path twice")
        for path in paths:
            if out.resolve() not in path.resolve().parents:
                raise _fail(what, f"files(): {path} is not inside the run's output directory")
        sink.open()
        for a in range(0, take, piece):
            sink.write(data[2 * a : 2 * min(take, a + piece)])
        sink.close()
        missing = [str(f.path) for f in sink.files() if not f.path.is_file()]
        if missing:
            raise _fail(what, f"after close(), {missing[0]} does not exist")
        _json(what, sink.info())
        line = sink.summary()
        if line is not None and not isinstance(line, str):
            raise _fail(what, "summary() must return a string or None")
        results.append({f.path.name: f.path.read_bytes() for f in sink.files()})
    if any(r != results[0] for r in results[1:]):
        raise _fail(what, "the files differ when the same stream arrives in pieces of another size")
    # abort() after a partial stream must not raise
    sink = cls(cls.Params.model_validate(params or {}), _context(cls, seed, n0, n, fs, base / "aborted"))
    sink.open()
    sink.write(data[: 2 * min(1000, sink.planned())])
    sink.abort()


def _json(what: str, info: Any) -> None:
    if not isinstance(info, dict):
        raise _fail(what, f"info() must return a dict, got {type(info).__name__}")
    try:
        json.dumps(info)
    except (TypeError, ValueError) as e:
        raise _fail(what, f"info() must be JSON-serialisable for run.json: {e}") from None


def _check_serving(
    what: str, cls: Any, params: dict[str, Any] | None, seed: int, n0: int, n: int, fs: int, base: Path
) -> None:
    if not issubclass(cls, ServingSink) or not issubclass(cls.Params, ServingParams):
        raise _fail(what, "a serving sink derives from ServingSink, its Params from ServingParams")
    port = _free_port()
    p = cls.Params.model_validate({**(params or {}), "host": "127.0.0.1", "port": port, "pacing": "unpaced"})
    sink = cls(p, _context(cls, seed, n0, n, fs, base / "serving"))
    ahead = RenderAhead(_Source(seed), n0, n, 16, 240_000)

    async def client() -> int:
        t_end = time.monotonic() + 20
        while True:
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                break
            except ConnectionRefusedError:
                if time.monotonic() > t_end:
                    raise _fail(what, f"serve() did not listen on port {port} within 20 s") from None
                await asyncio.sleep(0.02)
        got = 0
        while chunk := await reader.read(1 << 20):
            got += len(chunk)
        writer.close()
        return got

    async def main() -> int:
        serving = asyncio.create_task(sink.serve(ahead))
        reading = asyncio.create_task(client())
        done, _ = await asyncio.wait({serving, reading}, timeout=60, return_when=asyncio.FIRST_EXCEPTION)
        for task in (serving, reading):
            if task in done and task.exception() is not None:
                reading.cancel()
                serving.cancel()
                raise task.exception()  # ty: ignore[invalid-raise]
        if not serving.done() or not reading.done():
            serving.cancel()
            reading.cancel()
            raise _fail(
                what, f"serve() did not finish a stream of {n} samples (it must stop at the end of the stream)"
            )
        return reading.result()

    got = asyncio.run(main())
    if got == 0:
        raise _fail(what, f"a client received nothing of a stream of {n} samples")
    _json(what, sink.info())
