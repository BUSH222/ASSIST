"""Command-line interface: `assist`. Commands are added milestone by milestone (plan §8)."""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from assist import __version__

if TYPE_CHECKING:
    from assist.config.resolve import Resolved

app = typer.Typer(
    name="assist",
    help="ASSIST: synthetic 2.4 MS/s RTL-SDR IQ at 137 MHz, served over rtl_tcp or written to files.",
    no_args_is_help=True,
    add_completion=False,
)
elements_app = typer.Typer(help="Element sets (retlector.eu snapshots).", no_args_is_help=True)

ScenarioArg = Annotated[Path, typer.Argument(help="Scenario (.yaml, .yml or .json; a resolved one works too).")]
SetOpt = Annotated[
    list[str] | None, typer.Option("--set", help="Override a field: path=value, e.g. sources\\[z1].freq_hz=137.35e6.")
]
FromOpt = Annotated[str | None, typer.Option("--from", help="ISO time, `start`, or +offset (default: clock start).")]
ToOpt = Annotated[str | None, typer.Option("--to", help="ISO time or +offset from the clock start (default: end).")]
WorkersOpt = Annotated[int | None, typer.Option(help="Worker processes (default: the number of CPUs).")]


def _print_version(value: bool) -> None:
    if value:
        from assist import backend

        typer.echo(f"assist {__version__} (kernels: {backend.NAME})")
        raise typer.Exit()


def _register_subcommands() -> None:
    from assist.profile.cli import app as profile_app

    app.add_typer(profile_app, name="profile")
    app.add_typer(elements_app, name="elements")


_register_subcommands()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_print_version, is_eager=True, help="Show the version and exit."),
    ] = False,
) -> None:
    """ASSIST: Autonomous Satellite Signal Imitator Station."""


def _resolve(scenario: Path, sets: list[str] | None) -> Resolved:
    from pydantic import ValidationError

    from assist.config.loader import ConfigError
    from assist.config.resolve import resolve

    try:
        r = resolve(scenario, tuple(sets or ()))
    except (ConfigError, ValidationError, KeyError, ValueError, FileNotFoundError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2) from e
    for w in r.warnings:
        typer.echo(f"warning: {w}", err=True)
    return r


def _workers(n: int | None) -> int:
    import os

    if n is not None and n < 1:
        raise typer.BadParameter("--workers must be >= 1")
    return n if n is not None else os.cpu_count() or 1


def _window(r: Resolved, t_from: str | None, t_to: str | None) -> tuple[float, float]:
    from assist.timebase import parse_iso, parse_time

    t0 = parse_iso(r.scenario.clock.start)
    a = parse_time(t_from, t0) if t_from else t0
    b = parse_time(t_to, t0) if t_to else t0 + r.scenario.clock.duration
    if b <= a:
        raise typer.BadParameter("--to must be after --from")
    return a, b


@app.command()
def schema(
    out: Annotated[Path | None, typer.Option("-o", "--out", help="Write here instead of stdout.")] = None,
) -> None:
    """JSON Schema of a scenario (after band defaults and presets are applied)."""
    import json

    from assist.config.schema import Scenario

    text = json.dumps(Scenario.model_json_schema(), indent=1) + "\n"
    if out is None:
        sys.stdout.write(text)
    else:
        out.write_text(text)


@app.command()
def validate(scenario: ScenarioArg, set_: SetOpt = None) -> None:
    """Check a scenario: load, draw, validate and resolve it."""
    r = _resolve(scenario, set_)
    sc = r.scenario
    n_em = sum(len(s.emitters) for s in sc.sources)
    typer.echo(f"ok: {len(sc.sources)} sources, {n_em} emitters, start {sc.clock.start}, scenario hash {r.hash}")


@app.command(name="resolve")
def resolve_cmd(
    scenario: ScenarioArg,
    out: Annotated[Path | None, typer.Option("-o", "--out", help="resolved.json (default: stdout).")] = None,
    set_: SetOpt = None,
) -> None:
    """Write the resolved, self-contained scenario (canonical JSON) and print its hash."""
    r = _resolve(scenario, set_)
    if out is None:
        sys.stdout.write(r.pretty())
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(r.pretty())
    typer.echo(f"scenario hash {r.hash}", err=True)


@app.command()
def passes(
    scenario: ScenarioArg,
    t_from: FromOpt = None,
    t_to: ToOpt = None,
    out: Annotated[
        Path | None, typer.Option("-o", "--out", help="Write CSV here (default: a table on stdout).")
    ] = None,
    set_: SetOpt = None,
) -> None:
    """Pass table of all satellite sources."""
    import numpy as np

    from assist.engine.scene import Scene
    from assist.timebase import format_iso

    r = _resolve(scenario, set_)
    a, b = _window(r, t_from, t_to)
    scene = Scene(r.scenario)
    rows = []
    for s in scene.sources:
        if s.platform.fixed:
            continue
        for p in s.platform.passes(a, b):
            g = s.platform.geometry(np.array([p.aos, p.tca, p.los]))
            rows.append(
                [
                    s.id,
                    "" if s.platform.norad is None else str(s.platform.norad),
                    s.platform.name,
                    format_iso(p.aos, 1),
                    format_iso(p.tca, 1),
                    format_iso(p.los, 1),
                    f"{p.max_el_deg:.2f}",
                    f"{g.az_deg[0]:.1f}",
                    f"{g.az_deg[1]:.1f}",
                    f"{g.az_deg[2]:.1f}",
                    f"{p.los - p.aos:.0f}",
                ]
            )
    rows.sort(key=lambda x: (x[3], x[0]))
    head = [
        "source_id",
        "norad",
        "name",
        "aos_utc",
        "tca_utc",
        "los_utc",
        "max_el_deg",
        "aos_az",
        "tca_az",
        "los_az",
        "duration_s",
    ]
    if out is not None:
        import csv

        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(head)
            w.writerows(rows)
        typer.echo(f"{len(rows)} passes -> {out}", err=True)
        return
    widths = [max(len(h), *(len(x[i]) for x in rows)) for i, h in enumerate(head)]
    for line in [head, *rows]:
        typer.echo("  ".join(c.ljust(w) for c, w in zip(line, widths, strict=True)).rstrip())


@app.command()
def catalog(
    scenario: ScenarioArg,
    out: Annotated[Path, typer.Option("-o", "--out", help="catalog.tle or catalog.json")],
    sources_only: Annotated[bool, typer.Option(help="Only the scenario's sources, not the whole store.")] = False,
    set_: SetOpt = None,
) -> None:
    """Element sets ADRIFT is given: the store with leave-one-outs removed and synthetic in-catalog objects added."""
    from assist.orbits import catalog as cat

    r = _resolve(scenario, set_)
    es = cat.entries(r.scenario, Path.cwd(), sources_only)
    cat.write(es, out)
    typer.echo(f"{len(es)} element sets -> {out}", err=True)


@app.command()
def truth(
    scenario: ScenarioArg,
    out: Annotated[Path, typer.Option("-o", "--out", help="truth.csv / truth.jsonl; events go to <stem>.events.*")],
    t_from: FromOpt = None,
    t_to: ToOpt = None,
    step: Annotated[str, typer.Option(help="Row step, e.g. 1s, 100ms.")] = "1s",
    fmt: Annotated[str, typer.Option("--format", help="csv or jsonl")] = "csv",
    platforms: Annotated[
        str, typer.Option(help="Comma-separated platform kinds to include: satellite, ground, local.")
    ] = "satellite,ground,local",
    set_: SetOpt = None,
) -> None:
    """Truth for every emitter: frequencies, Doppler, geometry and levels per step, plus an event list (every burst
    with its frequency and strength)."""
    from assist import truth as tr
    from assist.engine.scene import Scene
    from assist.timebase import parse_duration

    if fmt not in ("csv", "jsonl"):
        raise typer.BadParameter("--format must be csv or jsonl")
    kinds = tuple(k.strip() for k in platforms.split(",") if k.strip())
    bad = [k for k in kinds if k not in tr.PLATFORMS]
    if bad or not kinds:
        raise typer.BadParameter(f"--platforms takes {', '.join(tr.PLATFORMS)} (got {platforms!r})")
    r = _resolve(scenario, set_)
    a, b = _window(r, t_from, t_to)
    table = tr.compute(Scene(r.scenario), a, b, parse_duration(step), kinds)
    write = tr.write_csv if fmt == "csv" else tr.write_jsonl
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        write(table.rows, tr.COLUMNS, f)
    msg = f"{len(table.rows)} rows -> {out}"
    if table.events:
        ev_path = out.with_name(f"{out.stem}.events.{fmt}")
        with ev_path.open("w", newline="") as f:
            write(table.events, tr.EVENT_COLUMNS, f)
        msg += f", {len(table.events)} events -> {ev_path}"
    typer.echo(msg, err=True)


@app.command()
def render(
    scenario: ScenarioArg,
    out: Annotated[Path, typer.Option("-o", "--out", help="Output directory (files + run.json).")],
    t_from: FromOpt = None,
    duration: Annotated[
        str | None, typer.Option(help="e.g. 90s, 16min, 2h (default: to the end of the scenario clock).")
    ] = None,
    fmt: Annotated[str, typer.Option("--format", help="wav (split every 10 min) or cu8 (one raw file).")] = "wav",
    rf64: Annotated[bool, typer.Option("--rf64", help="With wav: one RF64 file instead of 10 min splits.")] = False,
    force: Annotated[bool, typer.Option("--force", help="Overwrite existing output files.")] = False,
    workers: WorkersOpt = None,
    set_: SetOpt = None,
) -> None:
    """Render the station's uint8 IQ stream to files, in parallel worker processes (same bytes for any count)."""
    import time

    from assist.engine import workers as pool
    from assist.engine.control import RenderError
    from assist.engine.render import Renderer
    from assist.engine.scene import Scene
    from assist.sinks import runinfo
    from assist.sinks.wavfile import FileSink, plan_files
    from assist.timebase import format_iso, parse_duration, parse_iso, parse_time, sample_at

    nw = _workers(workers)
    if fmt not in ("wav", "cu8"):
        raise typer.BadParameter("--format must be wav or cu8")
    if rf64 and fmt != "wav":
        raise typer.BadParameter("--rf64 goes with --format wav")
    kind = "rf64" if rf64 else fmt
    r = _resolve(scenario, set_)
    sc = r.scenario
    t0 = parse_iso(sc.clock.start)
    a = parse_time(t_from, t0) if t_from else t0
    dur = parse_duration(duration) if duration else t0 + sc.clock.duration - a
    if dur <= 0:
        raise typer.BadParameter("nothing to render: the window is empty")
    try:
        rd = Renderer(Scene(sc))
    except (RenderError, ValueError, NotImplementedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2) from e
    n0, n = sample_at(a, rd.fs), round(dur * rd.fs)
    files = plan_files(out, kind, rd.fc_hz, rd.fs, sc.station.tz, n0, n)
    run_path = out / "run.json"
    existing = [p for p in [*(f.path for f in files), run_path] if p.exists()]
    if existing and not force:
        typer.echo(f"error: {existing[0]} exists (use --force to overwrite)", err=True)
        raise typer.Exit(1)
    nw = min(nw, len(pool.tasks(n0, n, rd.fs)))
    runinfo.write_run_json(run_path, runinfo.run_info(r, rd, kind, n0, n, files, "rendering", workers=nw))
    typer.echo(
        f"rendering {n / rd.fs:.1f} s from {format_iso(n0 / rd.fs, 3)} into {len(files)} file(s) in {out}"
        f" ({nw} worker{'s' if nw > 1 else ''})",
        err=True,
    )
    sink = FileSink(files, kind, rd.fs)
    tic = time.perf_counter()
    report = 60 * rd.fs
    stream = pool.iter_u8(r.canonical, Path.cwd(), n0, n, rd.fs, nw) if nw > 1 else rd.iter_u8(n0, n)
    try:
        for chunk in stream:
            sink.write(chunk)
            if sink.written >= report or sink.written == n:
                done_s = sink.written / rd.fs
                speed = done_s / (time.perf_counter() - tic)
                typer.echo(f"  {done_s:7.1f} / {n / rd.fs:.1f} s  ({speed:.1f}x real time)", err=True)
                report += 60 * rd.fs
        sink.close()
    except RenderError as e:
        sink.close_quietly()
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2) from e
    elapsed = time.perf_counter() - tic
    info = runinfo.run_info(r, rd, kind, n0, n, files, "complete", workers=nw, render_seconds=round(elapsed, 3))
    runinfo.write_run_json(run_path, info)
    for f in files:
        typer.echo(str(f.path))
    typer.echo(f"done: {n / rd.fs / elapsed:.1f}x real time, scenario hash {r.hash}", err=True)


@app.command()
def fft(
    scenario: ScenarioArg,
    out: Annotated[Path, typer.Option("-o", "--out", help="Output directory (PNG + run.json).")] = Path("out/fft"),
    t_from: FromOpt = None,
    duration: Annotated[
        str | None,
        typer.Option(help="Stream to plot, e.g. 90s, 1h (default: the fft sink's `duration`; one is required)."),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing PNG.")] = False,
    workers: WorkersOpt = None,
    set_: SetOpt = None,
) -> None:
    """Waterfall PNG of the stream, from the scenario's `fft` sink (width, row_seconds, duration, min_db, max_db)."""
    import time

    from assist.engine import workers as pool
    from assist.engine.control import RenderError
    from assist.engine.render import Renderer
    from assist.engine.scene import Scene
    from assist.sinks import runinfo
    from assist.sinks.fft import FftSpec, WaterfallSink, waterfall_name
    from assist.sinks.wavfile import OutFile
    from assist.timebase import format_iso, parse_iso, parse_time, sample_at

    nw = _workers(workers)
    r = _resolve(scenario, set_)
    sc = r.scenario
    cfg = next((s["fft"] or {} for s in sc.sinks if "fft" in s), None)
    if cfg is None:
        typer.echo("error: the scenario has no `fft` entry in `sinks:`", err=True)
        raise typer.Exit(2)
    try:
        spec = FftSpec.from_dict(cfg, duration)
        rd = Renderer(Scene(sc))
        sink = WaterfallSink(spec, rd.fs)
    except (RenderError, ValueError, NotImplementedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2) from e
    t0 = parse_iso(sc.clock.start)
    a = parse_time(t_from, t0) if t_from else t0
    n0, n = sample_at(a, rd.fs), sink.n_samples
    png = out / waterfall_name(rd.fc_hz, n0 / rd.fs, sc.station.tz)
    run_path = out / "run.json"
    existing = [p for p in (png, run_path) if p.exists()]
    if existing and not force:
        typer.echo(f"error: {existing[0]} exists (use --force to overwrite)", err=True)
        raise typer.Exit(1)
    files = [OutFile(png, 0, n, n0)]
    nw = min(nw, len(pool.tasks(n0, n, rd.fs)))
    runinfo.write_run_json(run_path, runinfo.run_info(r, rd, "fft", n0, n, files, "rendering", workers=nw))
    typer.echo(
        f"plotting {n / rd.fs:.1f} s from {format_iso(n0 / rd.fs, 3)}: {sink.width} x {sink.height} px, "
        f"{sink.row_samples / rd.fs:.3f} s per row ({nw} worker{'s' if nw > 1 else ''})",
        err=True,
    )
    tic = time.perf_counter()
    stream = pool.iter_u8(r.canonical, Path.cwd(), n0, n, rd.fs, nw) if nw > 1 else rd.iter_u8(n0, n)
    status, code = "complete", 0
    try:
        for chunk in stream:
            sink.write(chunk)
        sink.close()
    except KeyboardInterrupt:
        status = "stopped"
    except RenderError as e:
        status, code = "error", 2
        typer.echo(f"error: {e}", err=True)
    elapsed = time.perf_counter() - tic
    if sink.rows:
        sink.save(png)
    info = runinfo.run_info(
        r,
        rd,
        "fft",
        n0,
        n,
        files if sink.rows else [],
        status,
        workers=nw,
        render_seconds=round(elapsed, 3),
        fft=sink.info(),
    )
    runinfo.write_run_json(run_path, info)
    if sink.rows:
        typer.echo(str(png))
    if sink.min_db is not None:
        typer.echo(
            f"{status}: {sink.rows}/{sink.height} rows, levels {sink.min_db:.1f}..{sink.max_db:.1f} dB", err=True
        )
    else:
        typer.echo(f"{status}: {sink.rows}/{sink.height} rows (level not yet calibrated)", err=True)
    if code:
        raise typer.Exit(code)


@app.command()
def bench(
    scenario: ScenarioArg,
    t_from: FromOpt = None,
    duration: Annotated[str, typer.Option(help="Stream to render, e.g. 60s or 10min.")] = "60s",
    workers: WorkersOpt = None,
    out: Annotated[Path, typer.Option("-o", "--out", help="Directory for <scenario>.json.")] = Path("out/bench"),
    set_: SetOpt = None,
) -> None:
    """Render speed vs real time, on one worker (one core) and on N, without writing IQ (plan §4.5)."""
    import json

    from assist.engine import bench as bn
    from assist.engine.control import RenderError
    from assist.timebase import format_iso, parse_duration, parse_iso, parse_time, sample_at

    nw = _workers(workers)
    r = _resolve(scenario, set_)
    sc = r.scenario
    fs = int(sc.frontend.fs_hz)
    t0 = parse_iso(sc.clock.start)
    n0 = sample_at(parse_time(t_from, t0) if t_from else t0, fs)
    n = round(parse_duration(duration) * fs)
    if n <= 0:
        raise typer.BadParameter("--duration must be positive")
    try:
        found = bn.run(r, n0, n, nw, Path.cwd(), lambda m: typer.echo(m, err=True))
    except (RenderError, ValueError, NotImplementedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2) from e
    res: dict[str, Any] = {"scenario": str(scenario), "start_utc": format_iso(n0 / fs, 3), **found}
    one = res["one_worker"]
    typer.echo(
        f"engine: {res['engine']}; {res['emitters']} emitters; {res['duration_s']:.1f} s from {res['start_utc']}"
    )
    typer.echo(f"1 worker:  {one['x_real_time']:6.2f}x real time ({one['seconds']:.2f} s)")
    for k, v in one["stage_s_per_stream_s"].items():
        typer.echo(f"  {k:<9} {v * 1000:7.1f} ms per s of stream")
    typer.echo(f"  {'other':<9} {one['other_s_per_stream_s'] * 1000:7.1f} ms per s of stream")
    if "workers" in res:
        w = res["workers"]
        same = "identical" if w["identical_to_one_worker"] else "DIFFERENT"
        typer.echo(
            f"{w['count']} workers: {w['x_real_time']:6.2f}x real time ({w['seconds']:.2f} s; start-up and warm-up "
            f"{w['start_up_and_warm_up_s']:.1f} s); stream {same} to 1 worker"
        )
    path = out / f"{scenario.stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1) + "\n")
    typer.echo(str(path))
    if "workers" in res and not res["workers"]["identical_to_one_worker"]:
        raise typer.Exit(1)


def _starts_now(scenario: Path, sets: list[str] | None) -> bool:
    """Whether the scenario's clock starts `now` (a live stream); call after _resolve has accepted it."""
    from assist.config.loader import load_scenario

    clock = load_scenario(scenario, tuple(sets or ())).get("clock") or {}
    return isinstance(clock, dict) and str(clock.get("start", "now")) == "now"


def _rtl_tcp_sink(sinks: list[dict[str, Any]], host: str | None, port: int | None) -> tuple[str, int]:
    """Listen address: the options, else the scenario's `rtl_tcp` sink, else 127.0.0.1:1234."""
    cfg = next((s["rtl_tcp"] for s in sinks if isinstance(s.get("rtl_tcp"), dict)), {})
    return host or str(cfg.get("host") or "127.0.0.1"), port if port is not None else int(cfg.get("port", 1234))


@app.command()
def serve(
    scenario: ScenarioArg,
    host: Annotated[
        str | None, typer.Option(help="Listen address (default: the scenario's rtl_tcp sink, else 127.0.0.1).")
    ] = None,
    port: Annotated[
        int | None, typer.Option(help="TCP port (default: the scenario's rtl_tcp sink, else 1234).")
    ] = None,
    pacing: Annotated[str, typer.Option(help="realtime, or unpaced (as fast as the clients read).")] = "realtime",
    t_from: FromOpt = None,
    duration: Annotated[
        str | None, typer.Option(help="Stop after this much stream, e.g. 1h (default: endless).")
    ] = None,
    lookahead: Annotated[str, typer.Option(help="Render-ahead ring, e.g. 1s.")] = "1s",
    out: Annotated[Path, typer.Option("-o", "--out", help="Directory for run.json.")] = Path("out/serve"),
    set_: SetOpt = None,
) -> None:
    """Serve the station's uint8 IQ stream over rtl_tcp (read-only clients such as SDR++). Ctrl-C stops it.

    With `clock.start: now` and realtime pacing the stream is live: it starts at the next whole second after the
    pre-roll, and sample time equals wall time. Otherwise it starts at --from (default: the clock start).
    """
    import asyncio
    import logging
    import math
    import time

    from assist.engine.control import RenderError
    from assist.engine.render import Renderer
    from assist.engine.scene import Scene
    from assist.sinks import runinfo
    from assist.sinks.rtltcp import CHUNK_SAMPLES, PACINGS, RenderAhead, RtlTcpServer, live_start, stderr_handler
    from assist.timebase import format_iso, parse_duration, parse_iso, parse_time, sample_at

    if pacing not in PACINGS:
        raise typer.BadParameter("--pacing must be realtime or unpaced")
    look = parse_duration(lookahead)
    if look <= 0:
        raise typer.BadParameter("--lookahead must be positive")
    r = _resolve(scenario, set_)
    sc = r.scenario
    host_, port_ = _rtl_tcp_sink(sc.sinks, host, port)
    live = pacing == "realtime" and t_from is None and _starts_now(scenario, set_)
    try:
        rd = Renderer(Scene(sc), prefetch=True)
    except (RenderError, ValueError, NotImplementedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2) from e
    n = round(parse_duration(duration) * rd.fs) if duration else None
    if n is not None and n <= 0:
        raise typer.BadParameter("--duration must be positive")
    t0 = parse_iso(sc.clock.start)
    wall = live_start(time.time(), look) if live else None
    n0 = wall * rd.fs if wall is not None else sample_at(parse_time(t_from, t0) if t_from else t0, rd.fs)
    ahead = RenderAhead(rd, n0, n, max(1, math.ceil(look * rd.fs / CHUNK_SAMPLES)), piece=10 * rd.hop)
    server = RtlTcpServer(ahead, rd.fs, rd.fc_hz, host=host_, port=port_, pacing=pacing, wall_start=wall)
    cfg = {"host": host_, "port": port_, "pacing": pacing, "live": live, "lookahead_s": look}
    run_path = out / "run.json"
    runinfo.write_run_json(run_path, runinfo.run_info(r, rd, "rtl_tcp", n0, n, [], "serving", serve=cfg))
    typer.echo(f"serving from {format_iso(n0 / rd.fs, 3)}{' (live)' if live else ''}; run.json in {out}", err=True)
    logger = logging.getLogger("assist.serve")
    handler = stderr_handler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    status, code = "complete" if n is not None else "stopped", 0
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        status = "stopped"
    except RenderError as e:
        status, code = "error", 2
        typer.echo(f"error: {e}", err=True)
    except OSError as e:
        status, code = "error", 1
        typer.echo(f"error: {e}", err=True)
    finally:
        logger.removeHandler(handler)
    st = server.stats
    stream = {
        "samples": st.samples,
        "seconds": st.samples / rd.fs,
        "stalls": st.stalls,
        "stall_s": round(st.stall_s, 3),
        "clients": st.clients,
        "slow_disconnects": st.slow_disconnects,
    }
    runinfo.write_run_json(run_path, runinfo.run_info(r, rd, "rtl_tcp", n0, n, [], status, serve=cfg, stream=stream))
    typer.echo(
        f"{status}: {st.samples / rd.fs:.1f} s streamed, {st.stalls} stall(s), {st.clients} client(s), "
        f"scenario hash {r.hash}",
        err=True,
    )
    if code:
        raise typer.Exit(code)


def _fetch_feeds(feeds: str, out: Path) -> None:
    from assist.orbits.elements import fetch as do_fetch

    try:
        paths = do_fetch([f.strip() for f in feeds.split(",") if f.strip()], out)
    except (FileExistsError, ValueError, OSError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    for p in paths:
        typer.echo(str(p))


@app.command(name="fetch_omm")
def fetch_omm(
    feeds: Annotated[str, typer.Option(help="Comma-separated feeds: active, satnogs.")] = "active,satnogs",
    out: Annotated[Path, typer.Option("-o", "--out", help="Snapshot directory.")] = Path("elements"),
) -> None:
    """Download active_<date>.csv and satnogs_<date>.csv from retlector.eu into elements/ (network).

    The date is the UTC day of the request. Existing files are never overwritten.
    """
    _fetch_feeds(feeds, out)


@elements_app.command()
def fetch(
    feeds: Annotated[str, typer.Option(help="Comma-separated feeds: active, satnogs.")] = "active,satnogs",
    out: Annotated[Path, typer.Option("-o", "--out", help="Snapshot directory.")] = Path("elements"),
) -> None:
    """Same as `assist fetch_omm`."""
    _fetch_feeds(feeds, out)
