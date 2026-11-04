# ASSIST

**Autonomous Satellite Signal Imitator Station.** ASSIST synthesises the 2.4 MS/s IQ stream of a VHF station
(RTL-SDR v3 at 137.000 MHz) and serves it over rtl_tcp or writes it to files, so that
ADRIFT can be tested against a sky whose content is known exactly.

The implementation plan (`samples/plan.md`) and the per-milestone `checkpoint.md` are kept locally, not in git.

Status: M4 (live serving). Scenarios resolve to a self-contained document, the truth (frequencies, Doppler,
geometry, levels, events) can be computed, and `cw` emitters (zombies) render to the station's uint8 IQ stream with
profile-shaped noise. The stream is served over rtl_tcp in real time, or written as SDR++-style WAV or raw cu8 files.

## Usage

```sh
uv run assist validate scenarios/example.yaml                 # load, draw, validate, resolve
uv run assist resolve  scenarios/example.yaml -o out/resolved.json
uv run assist passes   scenarios/example.yaml --to +6h        # pass table of all satellite sources
uv run assist truth    scenarios/example.yaml -o out/truth.csv --to +2h [--step 1s] [--format jsonl]
uv run assist catalog  scenarios/example.yaml -o out/catalog.tle   # element sets ADRIFT is given
uv run assist render   scenarios/zombie_single.yaml -o out/m3      # WAV files (10 min each) + run.json
uv run assist render   scenarios/zombie_single.yaml -o out/raw --format cu8 --from +7min --duration 90s
uv run assist serve    scenarios/serve_demo.yaml                 # rtl_tcp on 127.0.0.1:1234, live, until Ctrl-C
uv run assist schema                                          # JSON Schema of a scenario
uv run assist elements fetch                                  # network: dated retlector.eu snapshots
uv run assist profile  build | labels | report                # data survey of samples/ (M1)
```

Scenarios are YAML or JSON (`--set path=value` overrides a field, e.g. `sources[z1].freq_hz=137.35e6`).
Renders are byte-identical for the same scenario, and any window equals the same slice of a longer render. WAV files
are named like SDR++ recordings (`baseband_137000000Hz_HH-MM-SS_DD-MM-YYYY.wav`, station local time); `--rf64` writes
one file instead of 10 min splits. The station position is read from `samples/station.yaml` when present; otherwise
the rounded position (55.5 N, 37.1 E, 172 m) is used.

`serve` speaks the rtl_tcp protocol (an R820T dongle header, then interleaved uint8 I/Q at 2.4 MS/s) to any number
of clients, e.g. SDR++'s rtl_tcp source; client commands (frequency, gain, ...) are logged and ignored, because the
tuning is fixed. A scenario whose clock starts `now` is served live, sample time equal to wall time; otherwise the
stream starts at `--from` (default: the clock start). The stream runs whether or not a client is connected, renders
about `--lookahead 1s` ahead, and stops after `--duration` if given; `--pacing unpaced` sends as fast as the clients
read. The port comes from `--port`, the scenario's `rtl_tcp` sink, or 1234.

## Development

Needs Python 3.15 (regular build, not 3.15t), [uv](https://docs.astral.sh/uv/), CMake ≥ 3.20 and a C++17 compiler
(Xcode command-line tools on macOS).

```sh
uv sync --all-groups        # installs everything and builds the C++ extension assist._dsp
uv run pre-commit install   # ruff, ruff-format, ty on commit
uv run pytest
```

- **C++ edits** under `cpp/` need no reinstall: the editable install rebuilds `assist._dsp` on the next import
  (a few seconds).
- **`ASSIST_PURE=1`** runs the numpy reference kernels (`assist.dsp_ref`) instead of the C++ ones.
- **`ASSIST_NATIVE=ON uv sync --reinstall-package assist`** builds with `-march=native` (local use only);
  a plain `uv sync --reinstall-package assist` switches back.

Large local data (recordings, waterfalls, element snapshots) lives in `samples/`, which is gitignored; tests never
need it. Tool outputs go to `out/`, one-off temporary files to `temp/` (both gitignored).
