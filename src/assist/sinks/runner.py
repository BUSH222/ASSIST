"""The runner behind `assist run` (PLAN.md R3): one workflow for every sink.

It picks the sinks (the scenario's `sinks:`, or those named with `--sink`), builds the Renderer once, writes `run.json`
before and after, and maps errors to exit codes. Then either

- **push sinks** (wav, cu8, fft, external ones): the stream [n0, n0 + n) is rendered as fast as it goes, on worker
  processes, and every piece is handed to each sink until that sink has what it planned; or
- **one serving sink** (rtl_tcp, external ones): a `RenderAhead` thread renders into a ring and the sink serves it,
  live when the scenario clock starts `now`.

A run is one or the other. A scenario that lists both kinds needs `--sink` to say which.
"""

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from assist import registry
from assist.engine import workers as pool
from assist.engine.control import RenderError
from assist.engine.render import Renderer
from assist.engine.scene import Scene
from assist.sinks import runinfo
from assist.sinks.base import ServingSink, Sink, SinkContext, sink_class
from assist.sinks.stream import CHUNK_SAMPLES, RenderAhead, live_start, stderr_handler
from assist.timebase import format_iso, parse_duration, parse_iso, parse_time, sample_at

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from assist.config.resolve import Resolved

PROGRESS_S = 60  # push runs report every this many seconds of stream


class RunError(Exception):
    """A run that cannot start or failed; `code` is the process exit code (2: scenario or render, 1: files, I/O)."""

    def __init__(self, message: str, code: int = 2) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RunOptions:
    out: Path
    sinks: tuple[str, ...] = ()  # --sink (empty: every sink the scenario lists)
    t_from: str | None = None
    duration: str | None = None
    workers: int = 1
    force: bool = False


def select_sinks(r: Resolved, names: tuple[str, ...]) -> list[tuple[type[Sink], dict[str, Any]]]:
    """(class, validated settings) of the sinks of a run: the named ones (one the scenario does not list runs with
    its defaults), else all the scenario lists. Push sinks only, or exactly one serving sink."""
    listed = {name: params for entry in r.scenario.sinks for name, params in entry.items()}
    if not names and not listed:
        known = ", ".join(registry.sinks.names())
        raise RunError(f"the scenario lists no sinks: add `sinks:` or name one with --sink (known: {known})")
    if len(set(names)) != len(names):
        raise RunError("--sink names a sink twice")
    chosen: list[tuple[type[Sink], dict[str, Any]]] = []
    for name in names or tuple(listed):
        try:
            cls = sink_class(name)
            params = listed[name] if name in listed else cls.Params().model_dump(mode="json")
        except ValueError as e:
            raise RunError(str(e)) from e
        chosen.append((cls, params))
    serving = [cls.kind for cls, _ in chosen if cls.serving]
    if serving and len(chosen) > 1:
        push = [cls.kind for cls, _ in chosen if not cls.serving]
        what = f"{', '.join(serving)} (serving) and {', '.join(push)} (files)" if push else ", ".join(serving)
        raise RunError(
            f"a run drives file sinks only or exactly one serving sink, not {what}: choose with --sink, "
            f"e.g. --sink {serving[0]}"
        )
    return chosen


def _describe(sinks: Sequence[tuple[Sink, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [{"kind": s.kind, "params": params, "info": s.info()} for s, params in sinks]


def run(r: Resolved, opt: RunOptions, echo: Callable[[str], None], say: Callable[[str], None]) -> int:
    """Run the scenario into its sinks. `echo` gets progress and messages (stderr), `say` the paths of the files
    written (stdout). Returns the exit code; a run that cannot start raises RunError."""
    chosen = select_sinks(r, opt.sinks)
    sc = r.scenario
    serving = chosen[0][0].serving
    try:
        rd = Renderer(Scene(sc), prefetch=serving)
    except (RenderError, ValueError, NotImplementedError) as e:
        raise RunError(str(e)) from e
    t0 = parse_iso(sc.clock.start)
    n = round(parse_duration(opt.duration) * rd.fs) if opt.duration else None
    if n is not None and n <= 0:
        raise RunError("--duration must be positive")
    if serving:
        return _serve(r, rd, chosen[0], opt, t0, n, echo)
    a = parse_time(opt.t_from, t0) if opt.t_from else t0
    if n is None:
        n = round((t0 + sc.clock.duration - a) * rd.fs)
    if n <= 0:
        raise RunError("nothing to render: the window is empty")
    return _push(r, rd, chosen, opt, sample_at(a, rd.fs), n, echo, say)


def _context(r: Resolved, rd: Renderer, kind: str, opt: RunOptions, n0: int, n: int | None, **kw: Any) -> SinkContext:
    scope = rd.scene.rng.scope("sink", kind)
    return SinkContext(rd.fs, rd.fc_hz, n0, n, opt.out, r.scenario.station.tz, scope, r.hash, **kw)


def _push(
    r: Resolved,
    rd: Renderer,
    chosen: list[tuple[type[Sink], dict[str, Any]]],
    opt: RunOptions,
    n0: int,
    n: int,
    echo: Callable[[str], None],
    say: Callable[[str], None],
) -> int:
    fs = rd.fs
    try:
        sinks = [
            (cls(cls.Params.model_validate(params), _context(r, rd, cls.kind, opt, n0, n)), params)
            for cls, params in chosen
        ]
    except ValueError as e:
        raise RunError(str(e)) from e
    need = {id(s): s.planned() for s, _ in sinks}
    if any(not 0 < k <= n for k in need.values()):
        raise RunError("a sink planned more samples than the stream has")
    n = max(need.values())  # the run ends when every sink has what it planned
    run_path = opt.out / "run.json"
    planned = [f for s, _ in sinks for f in s.files()]
    names = [f.path for f in planned]
    if len(set(names)) != len(names):
        raise RunError("two sinks would write the same file")
    existing = [p for p in [*names, run_path] if p.exists()]
    if existing and not opt.force:
        raise RunError(f"{existing[0]} exists (use --force to overwrite)", 1)
    nw = min(opt.workers, len(pool.tasks(n0, n, fs)))
    runinfo.write_run_json(
        run_path, runinfo.run_info(r, rd, _describe(sinks), n0, n, planned, "rendering", workers=nw)
    )
    kinds = ", ".join(s.kind for s, _ in sinks)
    echo(
        f"rendering {n / fs:.1f} s from {format_iso(n0 / fs, 3)} into {kinds} in {opt.out}"
        f" ({nw} worker{'s' if nw > 1 else ''})"
    )
    tic = time.perf_counter()
    report = PROGRESS_S * fs
    stream = pool.iter_u8(r.canonical, Path.cwd(), n0, n, fs, nw) if nw > 1 else rd.iter_u8(n0, n)
    status, code, done = "complete", 0, 0
    try:
        for s, _ in sinks:
            s.open()
        for chunk in stream:
            k = len(chunk) // 2
            for s, _ in sinks:
                take = min(k, need[id(s)] - done)
                if take > 0:
                    s.write(chunk if take == k else chunk[: 2 * take])
            done += k
            if done >= report or done == n:
                speed = done / fs / (time.perf_counter() - tic)
                echo(f"  {done / fs:7.1f} / {n / fs:.1f} s  ({speed:.1f}x real time)")
                report += PROGRESS_S * fs
        for s, _ in sinks:
            s.close()
    except KeyboardInterrupt:
        status = "stopped"
    except RenderError as e:
        status, code = "error", 2
        echo(f"error: {e}")
    except OSError as e:
        status, code = "error", 1
        echo(f"error: {e}")
    if status != "complete":
        for s, _ in sinks:
            s.abort()
    elapsed = time.perf_counter() - tic
    files = [f for s, _ in sinks for f in s.files()]
    info = runinfo.run_info(
        r, rd, _describe(sinks), n0, n, files, status, workers=nw, render_seconds=round(elapsed, 3)
    )
    runinfo.write_run_json(run_path, info)
    for f in files:
        say(str(f.path))
    for s, _ in sinks:
        line = s.summary()
        if line:
            echo(line)
    if status == "complete":
        echo(f"done: {n / fs / elapsed:.1f}x real time, scenario hash {r.hash}")
    else:
        echo(f"{status} after {done / fs:.1f} of {n / fs:.1f} s, scenario hash {r.hash}")
    return code


def _serve(
    r: Resolved,
    rd: Renderer,
    chosen: tuple[type[Sink], dict[str, Any]],
    opt: RunOptions,
    t0: float,
    n: int | None,
    echo: Callable[[str], None],
) -> int:
    """One serving sink: live (sample time = wall time, starting at the next whole second after the pre-roll) when
    the scenario clock starts `now`, pacing is realtime and no --from is given; else from --from or the clock start."""
    cls, params = chosen
    fs = rd.fs
    try:
        p = cls.Params.model_validate(params)
    except ValueError as e:
        raise RunError(str(e)) from e
    assert issubclass(cls, ServingSink)
    look, pacing = float(getattr(p, "lookahead", 1.0)), str(getattr(p, "pacing", "realtime"))
    live = pacing == "realtime" and opt.t_from is None and r.clock_now
    wall = live_start(time.time(), look) if live else None
    n0 = wall * fs if wall is not None else sample_at(parse_time(opt.t_from, t0) if opt.t_from else t0, fs)
    try:
        sink = cls(p, _context(r, rd, cls.kind, opt, n0, n, wall_start=wall))
    except ValueError as e:
        raise RunError(str(e)) from e
    assert isinstance(sink, ServingSink)
    ahead = RenderAhead(rd, n0, n, max(1, math.ceil(look * fs / CHUNK_SAMPLES)), piece=10 * rd.hop)
    run_path = opt.out / "run.json"
    described = [(sink, params)]
    runinfo.write_run_json(run_path, runinfo.run_info(r, rd, _describe(described), n0, n, [], "serving"))
    echo(f"serving ({cls.kind}) from {format_iso(n0 / fs, 3)}{' (live)' if live else ''}; run.json in {opt.out}")
    logger = logging.getLogger("assist.serve")
    handler = stderr_handler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    status, code = "complete" if n is not None else "stopped", 0
    try:
        asyncio.run(sink.serve(ahead))
    except KeyboardInterrupt:
        status = "stopped"
    except RenderError as e:
        status, code = "error", 2
        echo(f"error: {e}")
    except OSError as e:
        status, code = "error", 1
        echo(f"error: {e}")
    finally:
        logger.removeHandler(handler)
    runinfo.write_run_json(run_path, runinfo.run_info(r, rd, _describe(described), n0, n, [], status))
    echo(f"{status}: {sink.summary() or cls.kind}, scenario hash {r.hash}")
    return code
