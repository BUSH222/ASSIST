"""Command-line interface: `assist`."""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from assist import __version__

if TYPE_CHECKING:
    from assist.config.resolve import Resolved

app = typer.Typer(
    name="assist",
    help="ASSIST: synthetic 2.4 MS/s RTL-SDR IQ at 137 MHz, served over rtl_tcp or written to files (`assist run`).",
    no_args_is_help=True,
    add_completion=False,
)
elements_app = typer.Typer(help="Element sets (retlector.eu snapshots).", no_args_is_help=True)
plugins_app = typer.Typer(
    help="Signals, platforms and sinks: the core ones and those of plugins.", no_args_is_help=True
)

ScenarioArg = Annotated[Path, typer.Argument(help="Scenario (.yaml, .yml or .json; a resolved one works too).")]
SetOpt = Annotated[
    list[str] | None, typer.Option("--set", help="Override a field: path=value, e.g. sources\\[z1].freq_hz=137.35e6.")
]
FromOpt = Annotated[str | None, typer.Option("--from", help="ISO time, `start`, or +offset (default: clock start).")]
ToOpt = Annotated[str | None, typer.Option("--to", help="ISO time or +offset from the clock start (default: end).")]
PluginOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--plugin",
        help="Also load this external module (a .py file or a name), as if it were listed under `plugins:`. For "
        "experiments: a scenario should name its plugins itself.",
    ),
]
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
    app.add_typer(plugins_app, name="plugins")


_register_subcommands()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_print_version, is_eager=True, help="Show the version and exit."),
    ] = False,
) -> None:
    """ASSIST: Autonomous Satellite Signal Imitator Station."""


def _resolve(scenario: Path, sets: list[str] | None, extra_plugins: list[str] | None = None) -> Resolved:
    from pydantic import ValidationError

    from assist.config.loader import ConfigError
    from assist.config.resolve import resolve

    try:
        r = resolve(scenario, tuple(sets or ()), extra_plugins=tuple(extra_plugins or ()))
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
def validate(scenario: ScenarioArg, set_: SetOpt = None, plugin: PluginOpt = None) -> None:
    """Check a scenario: load, draw, validate and resolve it."""
    r = _resolve(scenario, set_, plugin)
    sc = r.scenario
    n_em = sum(len(s.emitters) for s in sc.sources)
    typer.echo(f"ok: {len(sc.sources)} sources, {n_em} emitters, start {sc.clock.start}, scenario hash {r.hash}")


@app.command(name="resolve")
def resolve_cmd(
    scenario: ScenarioArg,
    out: Annotated[Path | None, typer.Option("-o", "--out", help="resolved.json (default: stdout).")] = None,
    set_: SetOpt = None,
    plugin: PluginOpt = None,
) -> None:
    """Write the resolved, self-contained scenario (canonical JSON) and print its hash."""
    r = _resolve(scenario, set_, plugin)
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
    plugin: PluginOpt = None,
) -> None:
    """Pass table of all satellite sources."""
    import numpy as np

    from assist.engine.scene import Scene
    from assist.timebase import format_iso

    r = _resolve(scenario, set_, plugin)
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
    plugin: PluginOpt = None,
) -> None:
    """Element sets ADRIFT is given: the store with leave-one-outs removed and synthetic in-catalog objects added."""
    from assist.orbits import catalog as cat

    r = _resolve(scenario, set_, plugin)
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
        str | None,
        typer.Option(
            help="Comma-separated platforms to include, e.g. satellite,ground,local or one from a plugin "
            "(default: all)."
        ),
    ] = None,
    set_: SetOpt = None,
    plugin: PluginOpt = None,
) -> None:
    """Truth for every emitter: frequencies, Doppler, geometry and levels per step, plus an event list (every burst
    with its frequency and strength)."""
    from assist import truth as tr
    from assist.engine.scene import Scene
    from assist.timebase import parse_duration

    if fmt not in ("csv", "jsonl"):
        raise typer.BadParameter("--format must be csv or jsonl")
    r = _resolve(scenario, set_, plugin)  # first: the scenario's plugins may add platforms
    kinds = None
    if platforms is not None:
        from assist import registry

        known = registry.platforms.names()
        kinds = tuple(k.strip() for k in platforms.split(",") if k.strip())
        if not kinds or any(k not in known for k in kinds):
            raise typer.BadParameter(f"--platforms takes {', '.join(known)} (got {platforms!r})")
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
def run(
    scenario: ScenarioArg,
    sink: Annotated[
        list[str] | None,
        typer.Option(
            "--sink",
            help="Run only this sink (repeatable). Default: every sink in the scenario's `sinks:`. A sink the "
            "scenario does not list runs with its default settings.",
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("-o", "--out", help="Output directory: files and run.json (default: out/<scenario>)."),
    ] = None,
    t_from: FromOpt = None,
    duration: Annotated[
        str | None,
        typer.Option(
            help="Stream length, e.g. 90s, 16min, 2h (default: to the end of the scenario clock; a serving sink "
            "is endless)."
        ),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Overwrite existing output files.")] = False,
    workers: WorkersOpt = None,
    set_: SetOpt = None,
    plugin: PluginOpt = None,
) -> None:
    """Run the scenario into its sinks: IQ files (wav, cu8), a waterfall PNG (fft), or a server (rtl_tcp).

    Sinks and their settings come from the scenario's `sinks:` list; `--set 'sinks\\[wav].rf64=true'` changes one.
    File sinks render as fast as the machine allows, in parallel worker processes (same bytes for any count), and
    several can share a run. A serving sink runs alone, until Ctrl-C or --duration: with `clock.start: now` and
    realtime pacing the stream is live (sample time equals wall time), otherwise it starts at --from.
    """
    from assist.sinks import runner

    nw = _workers(workers)
    r = _resolve(scenario, set_, plugin)
    opt = runner.RunOptions(
        out if out is not None else Path("out") / scenario.stem, tuple(sink or ()), t_from, duration, nw, force
    )
    try:
        code = runner.run(r, opt, lambda m: typer.echo(m, err=True), typer.echo)
    except runner.RunError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(e.code) from e
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
    plugin: PluginOpt = None,
) -> None:
    """Render speed vs real time, on one worker (one core) and on N, without writing IQ (plan §4.5)."""
    import json

    from assist.engine import bench as bn
    from assist.engine.control import RenderError
    from assist.timebase import format_iso, parse_duration, parse_iso, parse_time, sample_at

    nw = _workers(workers)
    r = _resolve(scenario, set_, plugin)
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


def _module_detail(category: str, cls: Any) -> str:
    if category == "signal":
        return str(cls.mode)
    if category == "platform":
        return "fixed" if cls.fixed else "geometry"
    return "serving" if cls.serving else "push"


@plugins_app.command(name="list")
def plugins_list(
    scenario: Annotated[
        Path | None,
        typer.Argument(help="Show what this scenario would load (its `plugins:` list). Nothing is rendered."),
    ] = None,
    set_: SetOpt = None,
    plugin: PluginOpt = None,
) -> None:
    """Registered signals, platforms and sinks with their origin (core, or the plugin that provides them).

    Without a scenario (and without --plugin) only the core modules are listed: nothing else is ever loaded. With a
    scenario, its plugins are loaded and checked (found, importable, API version, name clashes); the rest of the
    scenario is not validated.
    """
    from assist import plugins, registry
    from assist.config.loader import ConfigError, load_scenario

    entries: list[Any] = []
    dirs: list[Path] = [Path.cwd()]
    try:
        if scenario is not None:
            listed = load_scenario(scenario, tuple(set_ or ())).get("plugins") or []
            if not isinstance(listed, list):
                raise ConfigError(f"plugins must be a list, got {listed!r}")
            entries += listed
            dirs.append(scenario.resolve().parent)
        entries += [plugins.from_cli(p) for p in plugin or ()]
        resolved = plugins.activate(entries, dirs)
    except (ConfigError, plugins.PluginError, FileNotFoundError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2) from e
    rows = [
        (cat, name, _module_detail(cat, reg.get(name)), reg.origin(name))
        for cat, reg in registry.ALL.items()
        for name in reg.names()
    ]
    w = max(len(r[1]) for r in rows)
    for cat in registry.ALL:
        typer.echo(f"{cat}s")
        for _, name, detail, origin in (r for r in rows if r[0] == cat):
            typer.echo(f"  {name.ljust(w)}  {detail.ljust(10)}  {origin}")
    typer.echo(f"plugins (assist.api {plugins.API_VERSION})")
    if not resolved:
        typer.echo("  none" if scenario is None and not plugin else "  none listed")
    for p, e in zip(plugins.active(), resolved, strict=True):
        what = ", ".join(f"{cat} {name}" for cat, name in p.registered)
        facts = [f"version {p.version or '-'}", f"api {p.api}"]
        if p.sha256 is not None:
            facts.append(f"sha256 {p.sha256[:12]}")
        if p.distribution is not None:
            facts.append(f"distribution {p.distribution}")
        typer.echo(f"  {p.label}  ({', '.join(facts)})")
        typer.echo(
            f"    {'path' if 'path' in e else 'module'}: {getattr(p.module, '__file__', None) or p.module.__name__}"
        )
        typer.echo(f"    registers: {what}")
