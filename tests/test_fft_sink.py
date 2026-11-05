"""FFT waterfall sink: validation, the maths against samples/waterfall.py's, chunk-size independence, `assist fft`."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from assist import waterfall as wf
from assist.cli import app
from assist.sinks.fft import FftSpec, WaterfallSink

ROOT = Path(__file__).resolve().parent.parent
FS = 2_400_000
runner = CliRunner()


def _noise(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, 2 * n, dtype=np.uint8)


def test_spec_requires_duration_and_validates() -> None:
    with pytest.raises(ValueError, match="needs a duration"):
        FftSpec.from_dict({})
    with pytest.raises(ValueError, match="needs a duration"):
        FftSpec.from_dict({"width": 1024, "row_seconds": 1})
    assert FftSpec.from_dict({}, "10s").duration == 10.0  # --duration alone is enough
    assert FftSpec.from_dict({"duration": "1min"}, "10s").duration == 10.0  # and wins over the sink's
    for bad in ({"width": 1000}, {"width": 32}, {"width": 2**21}, {"width": True}, {"row_seconds": 0}):
        with pytest.raises(ValueError):
            FftSpec.from_dict({**bad, "duration": 5})
    with pytest.raises(ValueError, match="both min_db and max_db"):
        FftSpec.from_dict({"min_db": 0, "duration": 5})
    with pytest.raises(ValueError, match="above"):
        FftSpec.from_dict({"min_db": 10, "max_db": 10, "duration": 5})
    with pytest.raises(ValueError, match="unknown"):
        FftSpec.from_dict({"heigth": 5, "duration": 5})


def test_geometry() -> None:
    s = FftSpec.from_dict({"width": 16384, "row_seconds": 1, "duration": 10})
    assert (s.avg_count(FS), s.row_samples(FS), s.height(FS)) == (146, 146 * 16384, 10 * FS // (146 * 16384))
    with pytest.raises(ValueError, match="shorter than one row"):
        WaterfallSink(FftSpec.from_dict({"row_seconds": 2, "duration": 1}), FS)


def test_matches_capture_script_maths() -> None:
    """Fixed levels, 16384 wide: pixels equal samples/waterfall.py's (assist.waterfall), whatever the piece size."""
    n_rows, lo = 3, -2.0
    spec = FftSpec.from_dict({"min_db": lo, "max_db": lo + 30, "row_seconds": 0.1, "duration": 0.5})
    sink = WaterfallSink(spec, FS)
    rs = sink.row_samples
    u8 = _noise(rs * sink.height)
    for i in range(0, len(u8), 2 * 99_991):  # awkward piece size
        sink.write(u8[i : i + 2 * 99_991])
    sink.close()
    assert sink.height >= n_rows and sink.image.shape == (sink.height, 16384)
    for i in range(sink.height):
        iq = wf.iq_from_u8(u8[2 * i * rs : 2 * (i + 1) * rs])
        want = wf.to_pixels(wf.power_db(wf.mean_power(iq)), lo)
        assert np.array_equal(sink.image[i], want)


def test_width_independent_scale() -> None:
    """White noise reads the same level in dB at every width (power is referred to 16384 points)."""
    levels = []
    for w in (1024, 16384):
        sink = WaterfallSink(
            FftSpec.from_dict({"width": w, "min_db": 0, "max_db": 60, "row_seconds": 0.2, "duration": 0.4}), FS
        )
        sink.write(_noise(sink.n_samples, 1))
        levels.append(float(sink.image.mean()))
    assert abs(levels[0] - levels[1]) < 2  # of 255


def test_auto_levels_follow_the_script_rule() -> None:
    spec = FftSpec.from_dict({"row_seconds": 1.0, "duration": 8})
    sink = WaterfallSink(spec, FS)
    u8 = _noise(sink.n_samples, 2)
    for i in range(0, len(u8), 2 * 1_000_003):
        sink.write(u8[i : i + 2 * 1_000_003])
    sink.close()
    rs = sink.row_samples
    cal = 5  # ceil(5 s / 0.9967 s)
    mean = np.mean([wf.mean_power(wf.iq_from_u8(u8[2 * i * rs : 2 * (i + 1) * rs])) for i in range(cal)], axis=0)
    want_min = float(wf.power_db(mean).mean() - 5)
    assert sink.min_db == pytest.approx(want_min, abs=1e-6)
    assert sink.max_db == pytest.approx(want_min + 30)
    assert sink.rows == sink.height  # the calibration rows are in the image
    assert np.array_equal(
        sink.image[0], wf.to_pixels(wf.power_db(wf.mean_power(wf.iq_from_u8(u8[: 2 * rs]))), want_min)
    )


def test_run_shorter_than_calibration() -> None:
    sink = WaterfallSink(FftSpec.from_dict({"row_seconds": 1.0, "duration": 2}), FS)
    sink.write(_noise(sink.n_samples, 3))
    sink.close()
    assert sink.rows == 2 and sink.min_db is not None


def test_incomplete_run_raises(tmp_path: Path) -> None:
    sink = WaterfallSink(FftSpec.from_dict({"min_db": 0, "max_db": 30, "row_seconds": 0.1, "duration": 0.3}), FS)
    sink.write(_noise(sink.row_samples + 10))
    with pytest.raises(ValueError, match="planned"):
        sink.close()
    sink.save(tmp_path / "w.png")  # what exists is still saved
    assert Image.open(tmp_path / "w.png").size == (16384, 1)


SCENARIO = """
version: 1
seed: 3
clock: {{start: "2026-11-04T12:00:00Z", duration: 1s}}
station: {{file: null, lat: 55.5, lon: 37.1, alt_m: 172}}
profile: profiles/vhf137
sinks:
{sinks}
sources:
  - id: z1
    preset: zombie
    orbit: {{synthetic_pass: {{aos: "-2min", max_el_deg: 30, direction: asc, side: west}}}}
    freq_hz: 137.2e6
    schedule: {{type: continuous}}
"""


def _scenario(tmp_path: Path, sinks: str) -> Path:
    p = tmp_path / "z.yaml"
    p.write_text(SCENARIO.format(sinks=sinks))
    return p


def test_fft_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    sc = _scenario(tmp_path, "  - fft: {width: 4096, row_seconds: 0.1, duration: 0.8s, min_db: 0, max_db: 40}")
    out = tmp_path / "out"
    res = runner.invoke(app, ["fft", str(sc), "-o", str(out), "--workers", "1"])
    assert res.exit_code == 0, res.output
    (png,) = out.glob("*.png")
    assert png.name == "waterfall_137000000Hz_15-00-00_04-11-2026.png"
    img = Image.open(png)
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "complete" and run["fft"]["levels"] == "given"
    assert img.size == (4096, run["fft"]["height"]) and img.mode == "L"
    assert run["render"]["n_samples"] == run["fft"]["height"] * run["fft"]["avg_count"] * 4096

    again = runner.invoke(app, ["fft", str(sc), "-o", str(out)])
    assert again.exit_code == 1 and "exists" in again.output
    before = png.read_bytes()
    forced = runner.invoke(app, ["fft", str(sc), "-o", str(out), "--force", "--workers", "2"])
    assert forced.exit_code == 0 and png.read_bytes() == before  # same pixels for any worker count


def test_fft_cli_duration_is_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    sc = _scenario(tmp_path, "  - fft: {width: 4096}")
    res = runner.invoke(app, ["fft", str(sc), "-o", str(tmp_path / "o")])
    assert res.exit_code == 2 and "duration" in res.output and not (tmp_path / "o").exists()
    ok = runner.invoke(app, ["fft", str(sc), "-o", str(tmp_path / "o"), "--duration", "2s", "--workers", "1"])
    assert ok.exit_code == 0, ok.output
    run = json.loads((tmp_path / "o" / "run.json").read_text())
    assert run["fft"]["levels"] == "calibrated" and run["fft"]["min_db"] is not None


def test_fft_cli_needs_a_sink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    sc = _scenario(tmp_path, "  - rtl_tcp: {port: 1234}")
    res = runner.invoke(app, ["fft", str(sc), "-o", str(tmp_path / "o"), "--duration", "1s"])
    assert res.exit_code == 2 and "no `fft`" in res.output
