# ASSIST

**Autonomous Satellite Signal Imitator Station.** ASSIST synthesises the 2.4 MS/s IQ stream of a VHF station
(RTL-SDR v3 at 137.000 MHz) and serves it over rtl_tcp or writes it to files, so that
ADRIFT can be tested against a sky whose content is known exactly.

The implementation plan (`samples/plan.md`) and the per-milestone `checkpoint.md` are kept locally, not in git.

Status: M7 (terrestrial environment; M7 and M8 swapped, front-end realism is M8). Scenarios resolve to a
self-contained document, the truth (frequencies, Doppler, geometry, levels, events) can be computed, and the station's
uint8 IQ stream is rendered with profile-shaped noise.
- Satellite signals: `cw` (zombies), METEOR LRPT (`oqpsk`, 72 ksym/s with CADU framing), ORBCOMM subscriber downlinks
  (`sdpsk`, 4800 sym/s) and gateway bursts (`oqpsk_burst`), and LoRa bursts (`lora`, SpaceBEE-style); presets
  `meteor_lrpt`, `orbcomm`, `spacebee`, `zombie`.
- Terrestrial signals (ground and local sources, no geometry): VDL bursts (`d8psk`, VDL Mode 2 on 136.975 MHz;
  `gfsk`, VDL Mode 4 style on 136.925 MHz), airband AM voice (`am`, cheap synthetic speech), the 138.1104 MHz
  carrier, USB noise humps (`noise_comb`) and birdies (`cw`, `noise_band`); presets `vdl_ground`, `vdl_aircraft`
  (each burst its own frequency offset and level), `atc_am`, `carrier_138`, `usb_humps`, `birdies`.
  `bands/vhf137/terrestrial_default.yaml` puts this station's environment (levels, rates, burst lengths measured
  from the recording and the waterfall days) into any scenario.

Payloads are random, so the signals are realistic at the spectrum and symbol level but not decodable. The stream is
served over rtl_tcp in real time, or written as SDR++-style WAV or raw cu8 files. The DSP runs in C++ kernels
(pocketfft): 20 carriers plus noise render at ~8x real time on one core, the recording's hour with its terrestrial
environment at ~9x, and files render in parallel worker processes (20-29x on 6 cores, the same bytes for any number
of workers).

## Usage

```sh
uv run assist validate scenarios/example.yaml                 # load, draw, validate, resolve
uv run assist resolve  scenarios/example.yaml -o out/resolved.json
uv run assist passes   scenarios/example.yaml --to +6h        # pass table of all satellite sources
uv run assist truth    scenarios/example.yaml -o out/truth.csv --to +2h [--step 1s] [--format jsonl]
uv run assist catalog  scenarios/example.yaml -o out/catalog.tle   # element sets ADRIFT is given
uv run assist render   scenarios/zombie_single.yaml -o out/m3      # WAV files (10 min each) + run.json, all CPUs
uv run assist render   scenarios/zombie_single.yaml -o out/raw --format cu8 --from +7min --duration 90s --workers 2
uv run assist bench    scenarios/bench_cw20.yaml [--duration 5min] # speed on 1 worker and on all CPUs, no files
uv run assist serve    scenarios/serve_demo.yaml                 # rtl_tcp on 127.0.0.1:1234, live, until Ctrl-C
uv run assist serve    scenarios/serve_satellites.yaml           # METEOR, ORBCOMM (+ gateway) and LoRa, live
uv run assist serve    scenarios/serve_terrestrial.yaml          # VDL, airband AM, 138 MHz carrier, humps, live
uv run assist serve    scenarios/terrestrial_hour.yaml           # the recording's hour (past date), paced
uv run assist truth    scenarios/terrestrial_hour.yaml -o out/truth.csv --to +10min --platforms ground
uv run assist schema                                          # JSON Schema of a scenario
uv run assist elements fetch                                  # network: dated retlector.eu snapshots
uv run assist profile  build | labels | report                # data survey of samples/ (M1)
uv run assist profile  terrestrial [--band]                   # VDL/AM/138 levels (M7) -> terrestrial_default.yaml
```

Scenarios are YAML or JSON (`--set path=value` overrides a field, e.g. `sources[z1].freq_hz=137.35e6`). Named
channels from `bands/vhf137/band.yaml` (ORBCOMM `S-1` .. `S-12`, `orbcomm_gateway`) can be used in a source's
`channels` list, e.g. `channels: [S-5, S-8]` on an `orbcomm` source. Add the station's terrestrial environment with
`include: [bands/vhf137/terrestrial_default.yaml]` (with `profile: profiles/vhf137`); override any of its sources by
id, e.g. `sources: [{id: usb-humps, comb: {enabled: false}}]`. Ground and local sources take a received level
(`level: {cn0_dbhz: ...}`) instead of an orbit; `per_event` draws a frequency offset and level for every event.
Renders are byte-identical for the same scenario (on the same backend), whatever the number of workers, and any
window equals the same slice of a longer render. WAV files
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
- **`ASSIST_PURE=1`** runs the numpy reference kernels (`assist.dsp_ref`) instead of the C++ ones (~1.3x real time
  on 20 carriers). The noise generator gives the same bits on both; the rest agrees to float rounding, so rendered
  bytes differ by at most one count, in a few per million.
- **C++ kernels** (`cpp/`): `synth` (overlap-add synthesis), `white`/`noise` (Philox4x32-10 + Box-Muller noise,
  shaped by overlap-save), `adc`; `random_bytes` (payloads), `shape` (pulse shaping), `chirp` (LoRa), `nco`
  (carrier phase and level), `conv_encode` and `ccsds_randomise` (METEOR framing); pocketfft is vendored in
  `cpp/third_party/`. Each has a numpy twin, compared in `tests/test_dsp_ref.py`.
- **`ASSIST_NATIVE=ON uv sync --reinstall-package assist`** builds with `-march=native` (local use only);
  a plain `uv sync --reinstall-package assist` switches back.

Large local data (recordings, waterfalls, element snapshots) lives in `samples/`, which is gitignored; CI tests
never need it (`tests/test_m6_local.py`, `tests/test_m7_local.py` and `tests/test_render_local.py` run only where it,
or a render, exists). Tool outputs go to `out/`, one-off temporary files to `temp/` (both gitignored).
