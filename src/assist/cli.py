"""Command-line interface: `assist`. Commands are added milestone by milestone (plan §8)."""

from typing import Annotated

import typer

from assist import __version__

app = typer.Typer(
    name="assist",
    help="ASSIST: synthetic 2.4 MS/s RTL-SDR IQ at 137 MHz, served over rtl_tcp or written to files.",
    no_args_is_help=True,
    add_completion=False,
)


def _print_version(value: bool) -> None:
    if value:
        from assist import backend

        typer.echo(f"assist {__version__} (kernels: {backend.NAME})")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_print_version, is_eager=True, help="Show the version and exit."),
    ] = False,
) -> None:
    """ASSIST: Autonomous Satellite Signal Imitator Station."""
