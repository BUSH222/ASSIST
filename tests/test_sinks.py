"""File sinks: WAV, RF64 and cu8 written and read back; SDR++ naming and splitting; run.json; `assist run`."""

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from assist.cli import app
from assist.profile.wavio import Recording, read_wav_info, sdrpp_name_info
from assist.sinks.wavfile import FileSink, plan_files, rf64_header, sdrpp_name, wav_header
from assist.timebase import parse_iso, sample_at

ROOT = Path(__file__).resolve().parent.parent
FS = 2_400_000
FC = 137e6
runner = CliRunner()


def _stream(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, 2 * n, dtype=np.uint8)


def _write(files, fmt: str, data: np.ndarray, piece: int = 70_001) -> None:
    sink = FileSink(files, fmt, FS)
    for i in range(0, len(data), 2 * piece):
        sink.write(data[i : i + 2 * piece])
    sink.close()


def test_sdrpp_name_is_local_time() -> None:
    t = parse_iso("2026-11-01T16:39:00Z")
    name = sdrpp_name(FC, t + 0.9, "+03:00", "wav")
    assert name == "baseband_137000000Hz_19-39-00_01-11-2026.wav"  # the recording's own name
    start, fc = sdrpp_name_info(name, "+03:00")
    assert fc == 137_000_000 and start == dt.datetime(2026, 11, 1, 16, 39, tzinfo=dt.UTC)


def test_wav_round_trip_with_10_min_style_split(tmp_path: Path) -> None:
    """Split every 1 s here (600 s by default): names, headers, sizes and every byte read back by the profile tool's
    reader."""
    t0 = parse_iso("2026-11-04T12:00:00Z") + 0.25
    n0, n = sample_at(t0, FS), round(2.5 * FS)
    files = plan_files(tmp_path, "wav", FC, FS, "+03:00", n0, n, split_s=1.0)
    assert [f.n_samples for f in files] == [FS, FS, FS // 2]
    assert [f.path.name for f in files] == [
        "baseband_137000000Hz_15-00-00_04-11-2026.wav",
        "baseband_137000000Hz_15-00-01_04-11-2026.wav",
        "baseband_137000000Hz_15-00-02_04-11-2026.wav",
    ]
    data = _stream(n)
    _write(files, "wav", data)
    got = []
    for f in files:
        info = read_wav_info(f.path)
        assert (info.fs, info.n_samples, info.data_offset, info.rf64, info.size_fields_wrong) == (
            FS,
            f.n_samples,
            44,
            False,
            False,
        )
        assert f.path.stat().st_size == 44 + 2 * f.n_samples
        rec = Recording(f.path)
        assert rec.start_unix == int(f.start_sample / FS)  # the name's 1 s resolution
        got.append(rec.u8(0, rec.n_samples))
    assert np.array_equal(np.concatenate(got), data)


def test_rf64_and_cu8_round_trip(tmp_path: Path) -> None:
    n0, n = sample_at(parse_iso("2026-11-04T12:00:00Z"), FS), 1_000_003
    data = _stream(n, 1)
    (rf,) = plan_files(tmp_path / "rf", "rf64", FC, FS, "+03:00", n0, n)
    _write([rf], "rf64", data)
    info = read_wav_info(rf.path)
    assert info.rf64 and info.n_samples == n and not info.size_fields_wrong
    assert np.array_equal(Recording(rf.path).u8(0, n), data)
    (raw,) = plan_files(tmp_path / "raw", "cu8", FC, FS, "+03:00", n0, n)
    _write([raw], "cu8", data)
    assert raw.path.suffix == ".cu8" and np.array_equal(np.fromfile(raw.path, dtype=np.uint8), data)


def test_headers() -> None:
    h = wav_header(1000, FS)
    assert len(h) == 44 and h[:4] == b"RIFF" and h[8:16] == b"WAVEfmt "
    assert int.from_bytes(h[40:44], "little") == 2000
    assert int.from_bytes(h[24:28], "little") == FS and int.from_bytes(h[34:36], "little") == 8
    with pytest.raises(ValueError, match="4 GiB"):
        wav_header(2**31, FS)
    assert rf64_header(2**31, FS)[:4] == b"RF64"


def test_sink_refuses_a_wrong_sample_count(tmp_path: Path) -> None:
    files = plan_files(tmp_path, "cu8", FC, FS, "+03:00", 0, 100)
    sink = FileSink(files, "cu8", FS)
    sink.write(_stream(50))
    with pytest.raises(ValueError, match="planned"):
        sink.close()
    sink = FileSink(files, "cu8", FS)
    with pytest.raises(ValueError, match="more samples"):
        sink.write(_stream(101))


SCENARIO = """
version: 1
seed: 3
clock: {start: "2026-11-04T12:00:00Z", duration: 0.25s}
station: {file: null, lat: 55.5, lon: 37.1, alt_m: 172}
profile: profiles/vhf137
sources:
  - id: z1
    preset: zombie
    orbit: {synthetic_pass: {aos: "-2min", max_el_deg: 30, direction: asc, side: west}}
    freq_hz: 137.2e6
    schedule: {type: continuous}
"""


def test_render_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    sc = tmp_path / "z.yaml"
    sc.write_text(SCENARIO)
    out = tmp_path / "out"
    res = runner.invoke(app, ["run", str(sc), "--sink", "wav", "-o", str(out)])
    assert res.exit_code == 0, res.output
    (wav,) = out.glob("*.wav")
    assert wav.name == "baseband_137000000Hz_15-00-00_04-11-2026.wav"
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "complete" and run["render"]["n_samples"] == 600_000
    assert run["render"]["start_sample_abs"] == sample_at(parse_iso("2026-11-04T12:00:00Z"), FS)
    assert run["files"][0]["name"] == wav.name and run["scenario"]["sources"][0]["id"] == "z1"
    assert run["sinks"] == [
        {"kind": "wav", "params": {"rf64": False}, "info": {"format": "wav", "samples_written": 600_000}}
    ]
    assert run["plugins"] == [] and "format" not in run["render"]
    assert len(run["scenario_hash"]) == 64 and run["noise"]["sigma_counts"] == pytest.approx(26.6868)
    x = Recording(wav).u8(0, 600_000).astype(np.float64) - 127.5
    assert 25 < x.std() < 29  # noise plus a weak carrier

    again = runner.invoke(app, ["run", str(sc), "--sink", "wav", "-o", str(out)])
    assert again.exit_code == 1 and "exists" in again.output
    before = wav.read_bytes()
    forced = runner.invoke(app, ["run", str(sc), "--sink", "wav", "-o", str(out), "--force"])
    assert forced.exit_code == 0 and wav.read_bytes() == before  # deterministic: same bytes again

    part = runner.invoke(app, ["run", str(sc), "--sink", "cu8", "-o", str(tmp_path / "raw"), "--from", "+0.1s"])
    assert part.exit_code == 0, part.output
    (raw,) = (tmp_path / "raw").glob("*.cu8")
    assert raw.read_bytes() == before[44 + 2 * 240_000 :]  # window independence through the CLI

    # Two file sinks in one run get the same stream; the settings of one are changed with --set
    both = tmp_path / "both"
    args = ["run", str(sc), "--sink", "wav", "--sink", "cu8", "-o", str(both), "--set", "sinks[wav].rf64=true"]
    res = runner.invoke(app, args)
    assert res.exit_code == 0, res.output
    (rf, cu) = sorted(both.glob("baseband_*"), key=lambda p: p.suffix, reverse=True)
    assert rf.suffix == ".wav" and rf.read_bytes()[:4] == b"RF64"
    assert cu.read_bytes() == before[44:] == rf.read_bytes()[-len(before) + 44 :]
    run2 = json.loads((both / "run.json").read_text())
    assert [s["kind"] for s in run2["sinks"]] == ["wav", "cu8"] and run2["sinks"][0]["params"] == {"rf64": True}
    assert len(run2["files"]) == 2 and run2["scenario_hash"] == run["scenario_hash"]  # sinks are not in the hash


def test_render_cli_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    sc = tmp_path / "z.yaml"
    sc.write_text(SCENARIO)
    none = runner.invoke(app, ["run", str(sc), "-o", str(tmp_path / "o")])
    assert none.exit_code == 2 and "lists no sinks" in none.output and "wav, cu8, fft, rtl_tcp" in none.output
    bad = runner.invoke(app, ["run", str(sc), "--sink", "mp3", "-o", str(tmp_path / "o")])
    assert bad.exit_code == 2 and "unknown sink 'mp3'" in bad.output
    bad = runner.invoke(
        app, ["run", str(sc), "--sink", "cu8", "-o", str(tmp_path / "o"), "--set", "sinks[cu8].rf64=1"]
    )
    assert bad.exit_code == 2 and "rf64" in bad.output
    twice = runner.invoke(app, ["run", str(sc), "--sink", "cu8", "--sink", "cu8", "-o", str(tmp_path / "o")])
    assert twice.exit_code == 2 and "twice" in twice.output
    empty = runner.invoke(app, ["run", str(sc), "--sink", "cu8", "-o", str(tmp_path / "o"), "--from", "+1s"])
    assert empty.exit_code == 2 and "window is empty" in empty.output
    assert not (tmp_path / "o").exists()
    for cmd in ("render", "fft", "serve"):
        assert runner.invoke(app, [cmd, str(sc)]).exit_code != 0  # the old commands are gone
    ppm = runner.invoke(app, ["run", str(sc), "--sink", "wav", "-o", str(tmp_path / "o"), "--set", "frontend.ppm=2"])
    assert ppm.exit_code == 2 and "only 0 is supported" in ppm.output
