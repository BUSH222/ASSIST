"""CLI skeleton."""

from importlib.metadata import entry_points

from typer.testing import CliRunner

from assist import __version__, backend
from assist.cli import app

runner = CliRunner()


def test_console_script_registered() -> None:
    (ep,) = entry_points(group="console_scripts", name="assist")
    assert ep.value == "assist.cli:app"


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"assist {__version__} (kernels: {backend.NAME})"


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output
