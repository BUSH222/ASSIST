"""`assist profile build | labels | report` (plan §8, §9)."""

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Data survey and profile extractor (local data in samples/).", no_args_is_help=True)

SamplesOpt = Annotated[Path, typer.Option("--samples", help="Directory with the local data.")]


def _one(samples: Path, pattern: str) -> Path:
    found = sorted(samples.glob(pattern))
    if len(found) != 1:
        raise typer.BadParameter(f"expected exactly one {pattern} in {samples}, found {len(found)}")
    return found[0]


@app.command()
def build(
    out: Annotated[Path, typer.Option("-o", "--out", help="Profile directory.")] = Path("profiles/vhf137"),
    samples: SamplesOpt = Path("samples"),
    wav: Annotated[Path | None, typer.Option(help="Recording (default: the one baseband_*.wav).")] = None,
    raw: Annotated[list[Path] | None, typer.Option(help="Waterfall .raw (repeat; default: all).")] = None,
    station: Annotated[Path | None, typer.Option(help="Station file (default: station.yaml).")] = None,
    elements: Annotated[list[Path] | None, typer.Option(help="Element CSVs (repeat).")] = None,
    annotations: Annotated[list[Path] | None, typer.Option(help="Annotation YAMLs (repeat).")] = None,
    saw: Annotated[Path | None, typer.Option(help="SAW S21 table (default: saw.csv).")] = None,
    work: Annotated[Path, typer.Option(help="Intermediates and event list.")] = Path("out/profile"),
) -> None:
    """Run P1-P12 and write the profile."""
    from assist.profile.build import build as run

    run(
        wav=wav or _one(samples, "baseband_*.wav"),
        raw_paths=raw or sorted(samples.glob("waterfall_*.raw")),
        station_path=station or samples / "station.yaml",
        element_paths=list(elements or [samples / "active.csv", samples / "satnogs.csv"]),
        annotation_paths=list(annotations or [samples / "annotations.yaml"]),
        saw_path=saw or samples / "saw.csv",
        out_dir=out,
        work_dir=work,
        log=lambda m: typer.echo(m, err=True),
    )


@app.command()
def labels(
    profile: Annotated[Path, typer.Argument(help="Profile directory.")] = Path("profiles/vhf137"),
    out: Annotated[Path, typer.Option("-o", "--out")] = Path("out/labels"),
    samples: SamplesOpt = Path("samples"),
    n: Annotated[int, typer.Option(help="Events to sample (stratified by auto label).")] = 200,
) -> None:
    """Write a label sheet (events.csv + crop images) for optional manual labelling (plan §9.1)."""
    import json

    from assist.profile import labels as lb
    from assist.profile.rawio import open_raw

    prof = json.loads((profile / "profile.json").read_text())
    events = lb.read_events(Path(prof["events"]["file"]))
    pick = lb.stratified_sample(events, n)
    raws = {p.name: open_raw(p) for p in samples.glob("waterfall_*.raw")}
    rows = []
    for e in pick:
        crop = out / "crops" / f"{e['id']}.png"
        lb.crop_png(
            raws[e["file"]], int(e["row_start"]), int(e["row_end"]), float(e["f_lo_hz"]), float(e["f_hi_hz"]), crop
        )
        rows.append({**e, "crop": str(crop.relative_to(out)), "label": ""})
    import csv

    out.mkdir(parents=True, exist_ok=True)
    with (out / "events.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[*lb.FIELDS, "crop", "label"])
        w.writeheader()
        w.writerows(rows)
    (out / "VOCABULARY.txt").write_text("\n".join(lb.VOCABULARY) + "\n")
    typer.echo(f"{len(rows)} events -> {out / 'events.csv'} (fill the 'label' column; vocabulary in VOCABULARY.txt)")


@app.command()
def report(
    profile: Annotated[Path, typer.Argument(help="Profile directory.")] = Path("profiles/vhf137"),
    out: Annotated[Path, typer.Option("-o", "--out")] = Path("out/profile_report"),
    work: Annotated[Path, typer.Option(help="Intermediates written by build.")] = Path("out/profile"),
) -> None:
    """Plots and a Markdown summary of a profile (needs the profile extra: matplotlib)."""
    from assist.profile.report import write_report

    path = write_report(profile, out, work)
    typer.echo(f"report: {path}")
