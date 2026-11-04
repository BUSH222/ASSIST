"""Chunk-parallel rendering (plan §4.5) and `bench`: worker processes give the same bytes as one process.

The pool spawns fresh interpreters (a few seconds each in CI), so the windows are short and tasks are 1 s.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from assist.cli import app
from assist.config.loader import load_scenario
from assist.config.resolve import resolve_document
from assist.engine import workers
from assist.engine.render import Renderer
from assist.engine.scene import Scene
from assist.timebase import parse_iso, sample_at

ROOT = Path(__file__).resolve().parent.parent
FS = 2_400_000
runner = CliRunner()


@pytest.fixture(autouse=True)
def _cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)


@pytest.fixture(scope="module")
def scenario(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A keyed zombie whose on interval starts inside the window (rising edge), plus a continuous carrier."""
    doc = {
        "seed": 3,
        "clock": {"start": "2026-11-04T12:00:00Z", "duration": "10min"},
        "station": {"file": None, "lat": 55.5, "lon": 37.1, "alt_m": 172},
        "profile": "profiles/vhf137",
        "sources": [
            {
                "id": "z1",
                "preset": "zombie",
                "orbit": {"synthetic_pass": {"aos": "-3min", "max_el_deg": 50, "direction": "desc", "side": "east"}},
                "freq_hz": 137.35e6,
                "schedule": {"phase_s": 62.5},  # on from 12:00:02.5 UTC (1793793600 = 60 mod 220)
            },
            {
                "id": "c1",
                "preset": "zombie",
                "orbit": {"synthetic_pass": {"aos": "-2min", "max_el_deg": 30, "direction": "asc", "side": "west"}},
                "freq_hz": 136.1e6,
                "schedule": {"type": "continuous"},
            },
        ],
    }
    p = tmp_path_factory.mktemp("workers") / "w.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


def test_tasks_cut_at_absolute_multiples() -> None:
    fs = 10
    assert workers.tasks(25, 50, fs, task_s=2) == [(25, 15), (40, 20), (60, 15)]
    assert workers.tasks(40, 20, fs, task_s=2) == [(40, 20)]
    assert workers.tasks(40, 0, fs, task_s=2) == []
    parts = workers.tasks(123_457, 9 * FS, FS)
    assert sum(k for _, k in parts) == 9 * FS and all((a + k) % (4 * FS) == 0 for a, k in parts[:-1])


def test_pool_bytes_equal_one_process(scenario: Path) -> None:
    """Three workers, 1 s tasks starting mid-chunk: every task computes its lead-in block itself."""
    r = resolve_document(load_scenario(scenario), ROOT)
    n0 = sample_at(parse_iso("2026-11-04T12:00:00Z"), FS) + 777_777
    n = round(4.5 * FS)
    rd = Renderer(Scene(r.scenario))
    one = rd.u8(n0, n)
    z = rd.emitters[0]
    assert z.id == "z1/carrier" and z.stream(n0 // FS) is None and z.stream(n0 // FS + 2) is not None  # rising edge
    parts = list(workers.iter_u8(r.canonical, ROOT, n0, n, FS, 3, task_s=1))
    assert len(parts) == 5
    np.testing.assert_array_equal(np.concatenate(parts), one)


def test_render_workers_cli(scenario: Path, tmp_path: Path) -> None:
    files = {}
    for nw in (1, 3):
        out = tmp_path / f"w{nw}"
        res = runner.invoke(
            app,
            ["render", str(scenario), "-o", str(out), "--format", "cu8", "--duration", "9s", "--workers", str(nw)],
        )
        assert res.exit_code == 0, res.output
        run = json.loads((out / "run.json").read_text())
        assert run["status"] == "complete" and run["workers"] == nw and run["engine"].endswith("(M6)")
        (f,) = out.glob("*.cu8")
        files[nw] = f.read_bytes()
    assert len(files[1]) == 2 * 9 * FS
    assert files[1] == files[3]


def test_render_workers_must_be_positive(scenario: Path, tmp_path: Path) -> None:
    res = runner.invoke(app, ["render", str(scenario), "-o", str(tmp_path), "--duration", "1s", "--workers", "0"])
    assert res.exit_code != 0


def test_bench_cli(scenario: Path, tmp_path: Path) -> None:
    res = runner.invoke(app, ["bench", str(scenario), "--duration", "2s", "--workers", "2", "-o", str(tmp_path)])
    assert res.exit_code == 0, res.output
    data = json.loads((tmp_path / "w.json").read_text())
    assert data["emitters"] == 2 and data["duration_s"] == 2.0
    one = data["one_worker"]
    assert one["x_real_time"] > 0 and set(one["stage_s_per_stream_s"]) >= {"control", "synth", "noise", "adc"}
    assert data["workers"]["count"] == 2 and data["workers"]["identical_to_one_worker"] is True
    assert "x real time" in res.output
