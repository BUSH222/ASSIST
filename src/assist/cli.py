"""Command-line interface: `assist`. Commands are added milestone by milestone (plan §8)."""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

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
    set_: SetOpt = None,
) -> None:
    """Truth for every emitter: frequencies, Doppler, geometry and levels per step, plus an event list."""
    from assist import truth as tr
    from assist.engine.scene import Scene
    from assist.timebase import parse_duration

    if fmt not in ("csv", "jsonl"):
        raise typer.BadParameter("--format must be csv or jsonl")
    r = _resolve(scenario, set_)
    a, b = _window(r, t_from, t_to)
    table = tr.compute(Scene(r.scenario), a, b, parse_duration(step))
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
    set_: SetOpt = None,
) -> None:
    """Render the station's uint8 IQ stream to files (Python reference engine)."""
    import time

    from assist.engine.control import RenderError
    from assist.engine.render import Renderer
    from assist.engine.scene import Scene
    from assist.sinks import runinfo
    from assist.sinks.wavfile import FileSink, plan_files
    from assist.timebase import format_iso, parse_duration, parse_iso, parse_time, sample_at

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
    runinfo.write_run_json(run_path, runinfo.run_info(r, rd, kind, n0, n, files, "rendering"))
    typer.echo(
        f"rendering {n / rd.fs:.1f} s from {format_iso(n0 / rd.fs, 3)} into {len(files)} file(s) in {out}", err=True
    )
    sink = FileSink(files, kind, rd.fs)
    tic = time.perf_counter()
    report = 60 * rd.fs
    try:
        for chunk in rd.iter_u8(n0, n):
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
    info = runinfo.run_info(r, rd, kind, n0, n, files, "complete", render_seconds=round(elapsed, 3))
    runinfo.write_run_json(run_path, info)
    for f in files:
        typer.echo(str(f.path))
    typer.echo(f"done: {n / rd.fs / elapsed:.1f}x real time, scenario hash {r.hash}", err=True)


@elements_app.command()
def fetch(
    feeds: Annotated[str, typer.Option(help="Comma-separated feeds: active, satnogs.")] = "active,satnogs",
    out: Annotated[Path, typer.Option("-o", "--out", help="Snapshot directory.")] = Path("samples/elements"),
) -> None:
    """Download dated snapshots from retlector.eu (network). Existing files are never overwritten."""
    from assist.orbits.elements import fetch as do_fetch

    try:
        paths = do_fetch([f.strip() for f in feeds.split(",") if f.strip()], out)
    except (FileExistsError, ValueError, OSError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    for p in paths:
        typer.echo(str(p))
