# ASSIST

**Autonomous Satellite Signal Imitator Station.** ASSIST synthesises the IQ stream of a VHF station (an RTL-SDR v3
tuned to 137.000 MHz, 2.4 MS/s, unsigned 8-bit I/Q) and either serves it over **rtl_tcp** in real time or writes it to
files, so that satellite-tracking software can be tested against a sky whose content is known exactly.

A *scenario* (a YAML file) describes the sky: which satellites are overhead and what they transmit, which aircraft and
ground stations are on the air, what interference the receiver itself makes, and what the noise floor looks like.
From one scenario ASSIST can produce:

- the **IQ stream** (`serve`, `render`, `fft`), and
- the **truth** (`truth`, `passes`, `catalog`): the exact frequency, Doppler shift, geometry and level of every signal
  at every moment, and the element sets a tracker should be given.

The same scenario and seed always give the same bytes, whatever the number of worker processes, and any time window of a
render equals the same slice of a longer one.

What it can synthesise:

| Family | Signals |
|---|---|
| Satellites | METEOR-M LRPT (`oqpsk`, 72 ksym/s, CADU framing), ORBCOMM subscriber downlinks (`sdpsk`, 4800 sym/s) and gateway bursts (`oqpsk_burst`), LoRa bursts (`lora`, SpaceBEE-style), unmodulated carriers of "zombie" satellites (`cw`) with Doppler, per-pass oscillator offset and drift, optional tumble |
| Aircraft and ground | VDL Mode 2 (`d8psk`) and VDL Mode 4 (`gfsk`) bursts, airband AM voice (`am`), carriers |
| Inside the receiver | birdies (`cw`, `noise_band`), USB noise humps (`noise_comb`), measured noise floor |

Payloads are random: the signals are realistic at the spectrum and symbol level, but they are not decodable.

## Contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [Commands](#commands)
4. [Writing scenario files](#writing-scenario-files)
   - [Anatomy of a scenario](#anatomy-of-a-scenario)
   - [Top-level keys](#top-level-keys)
   - [Sources](#sources)
   - [Orbits](#orbits)
   - [Emitters](#emitters)
   - [Signal kinds and their parameters](#signal-kinds-and-their-parameters)
   - [Schedules](#schedules)
   - [Levels](#levels)
   - [Per-event draws](#per-event-draws)
   - [Presets](#presets)
   - [Includes, merging and overrides](#includes-merging-and-overrides)
   - [Random expressions](#random-expressions)
   - [Sinks](#sinks)
   - [Bands and profiles](#bands-and-profiles)
5. [The example scenarios](#the-example-scenarios)
6. [Recipes](#recipes)
7. [Outputs](#outputs)
8. [Element sets](#element-sets)
9. [Development](#development)

## Installation

Needs Python 3.15 (the regular build, not 3.15t), [uv](https://docs.astral.sh/uv/), CMake ≥ 3.20 and a C++17 compiler
(Xcode command-line tools on macOS).

```sh
uv sync --all-groups        # installs everything and builds the C++ extension assist._dsp
uv run assist --version     # prints the version and the kernels in use
```

All commands below are run from the repository root as `uv run assist ...`. Paths in a scenario (`profile`,
`elements.files`, `include`) are relative to the directory you run `assist` from, except `include`, which is relative to
the including file first and to the package second.

## Quick start

Serve the live "what my station sees" scenario to SDR++ (or any rtl_tcp client):

```sh
uv run assist serve scenarios/realtime.yaml --lookahead 4s
```

In SDR++ choose the **RTL-TCP** source, host `127.0.0.1`, port `1234`, sample rate `2.4 MHz`. The stream is fixed at
137.000 MHz; tuning and gain commands from the client are logged and ignored.

Check a scenario and look at its satellite passes:

```sh
uv run assist validate scenarios/realtime.yaml
uv run assist passes   scenarios/realtime.yaml --to +6h
```

Render ten minutes to a raw file and write the truth for the same window:

```sh
uv run assist render scenarios/signal_tour.yaml -o out/tour --format cu8
uv run assist truth  scenarios/signal_tour.yaml -o out/tour/truth.csv
```

## Commands

Every command that reads a scenario takes the scenario path first and accepts `--set path=value` (repeatable) to
override a field without editing the file; see [overrides](#includes-merging-and-overrides). A time option
(`--from`, `--to`) is an ISO 8601 time with a zone (`2026-11-03T09:00:00Z`), `start`, or an offset from the scenario's
clock start (`+7min`, `+90s`, `+2h`). A duration is `90s`, `10min`, `2h`, `1d` or a number of seconds.

| Command | What it does |
|---|---|
| `validate SCENARIO` | Load, draw random values, validate and resolve. Prints the number of sources and emitters and the scenario hash. Exit code 2 on an error. |
| `resolve SCENARIO [-o FILE]` | Write the **resolved** scenario: a self-contained canonical JSON in which every random draw, catalog element set, channel name, calibrated level and synthetic pass is filled in. A resolved file is itself a valid scenario and resolves to itself. |
| `passes SCENARIO [--from T] [--to T] [-o CSV]` | Pass table (AOS, TCA, LOS, max elevation, azimuths, duration) of every satellite source. Without `-o`, a table on stdout. |
| `truth SCENARIO -o FILE [--from T] [--to T] [--step 1s] [--format csv\|jsonl] [--platforms satellite,ground,local]` | Truth per step for every emitter, plus an event list (every burst with its frequency and strength). See [Outputs](#outputs). |
| `catalog SCENARIO -o FILE [--sources-only]` | The element sets the tracker under test is given (`.tle` or `.json`): the whole element store, with sources marked `in_catalog: false` removed and synthetic sources with `in_catalog: true` added. `--sources-only` limits it to the scenario's sources. |
| `render SCENARIO -o DIR [--from T] [--duration D] [--format wav\|cu8] [--rf64] [--force] [--workers N]` | Write the IQ stream to files plus `run.json`, in parallel worker processes (default: all CPUs). Default window: the whole scenario clock. |
| `fft SCENARIO [-o DIR] [--from T] [--duration D] [--force] [--workers N]` | A waterfall PNG of the stream from the scenario's `fft` [sink](#sinks). |
| `serve SCENARIO [--host H] [--port P] [--pacing realtime\|unpaced] [--from T] [--duration D] [--lookahead 1s] [-o DIR]` | Serve the stream over rtl_tcp until Ctrl-C (or `--duration`). |
| `bench SCENARIO [--from T] [--duration 60s] [--workers N] [-o DIR]` | Render speed relative to real time on one worker and on N workers, without writing IQ. Also checks that both give identical streams. |
| `schema [-o FILE]` | The JSON Schema of a resolved scenario. |
| `fetch_omm [--feeds active,satnogs] [-o DIR]` | **Network.** Download `active_<date>.csv` and `satnogs_<date>.csv` from retlector.eu into `elements/`. Existing files are never overwritten. (`elements fetch` is the same command.) |
| `profile build \| labels \| report \| terrestrial` | Extract the noise profile and the terrestrial environment from your own recordings (see [Bands and profiles](#bands-and-profiles)). |

Notes on the individual commands:

- **`serve`** speaks the rtl_tcp protocol (an R820T dongle header, then interleaved uint8 I/Q at 2.4 MS/s) to any
  number of clients. A scenario whose clock starts `now` is served *live*: sample time equals wall time, so the sky
  is the real one. Otherwise the stream starts at `--from` (default: the clock start) and is paced at real time.
  The stream runs whether or not a client is connected. `--pacing unpaced` sends as fast as the clients read. The
  address comes from `--host/--port`, then the scenario's `rtl_tcp` sink, then `127.0.0.1:1234`. The renderer works
  `--lookahead` ahead of the wall clock; scenarios with many sources need a few seconds of head start at launch
  (`--lookahead 4s`), after which the stream runs on time. Session statistics (stalls, clients) go to `out/serve/run.json`.
- **`render`** with `--format wav` writes SDR++-style files named `baseband_137000000Hz_HH-MM-SS_DD-MM-YYYY.wav`
  (station local time, `station.tz`), split every 10 minutes; `--rf64` writes one file instead. `--format cu8` writes
  one raw file of interleaved unsigned 8-bit I/Q (for SDR++'s "Better File Source" plugin, or any raw reader). The stream
  is 4.8 MB/s, 17 GB/hour. Existing files are not overwritten without `--force`.
- **`--workers`** only changes speed, never the bytes.
- **`ASSIST_PURE=1`** runs the numpy reference kernels instead of the C++ ones (about 1.3x real time on 20 carriers
  instead of ~8x per core). The two agree to float rounding: rendered bytes differ by at most one count, in a few per million.

## Writing scenario files

### Anatomy of a scenario

A scenario is a YAML (or JSON) mapping. Unknown keys are errors, so typos fail loudly. YAML numbers follow YAML 1.2
(`137.9e6` is a number); times stay strings, so quote them.

```yaml
version: 1                          # always 1
seed: 42                            # every random draw derives from it
clock: {start: now, duration: 2h}
station: {lat: 55.5, lon: 37.1, alt_m: 172}
profile: profiles/vhf137            # measured noise floor and calibrations
elements:
  files: [elements/active_2026-11-02.csv, elements/satnogs_2026-11-02.csv]
include: [bands/vhf137/terrestrial_default.yaml]    # sources from another file
sinks:
  - rtl_tcp: {port: 1234}
sources:
  - id: m2-4                        # a satellite from the catalog
    preset: meteor_lrpt
    orbit: {norad: 59051}
  - id: z1                          # a zombie on a synthetic pass
    preset: zombie
    orbit: {synthetic_pass: {aos: +2min, max_el_deg: 45}}
    freq_hz: "uniform(136.1e6, 136.9e6)"
```

Resolving a scenario turns it into the document the engine uses: `now` becomes the current UTC second, random
expressions are drawn, element sets are looked up, synthetic passes are solved, channel names become frequencies, and
every level becomes a transmit power (satellites) or a received power (ground and local). `assist resolve` shows the result.

### Top-level keys

| Key | Default | Meaning |
|---|---|---|
| `version` | `1` | Schema version; must be `1`. |
| `seed` | `0` | Non-negative integer. Every random draw is keyed by the seed and by *names* (source id, emitter id, absolute time), never by list position: adding or removing a source does not change any other source's draws. |
| `clock.start` | `now` | ISO 8601 time **with a zone** (`"2026-11-01T16:39:00Z"`), or `now` (the current UTC second at resolve time). |
| `clock.duration` | `3600` | Length of the scenario in seconds, or `90s`, `10min`, `2h`, `1d`. It is the default window of `render`, `truth` and `passes`; `serve` runs past it. |
| `station.lat`, `station.lon`, `station.alt_m` | 55.5, 37.1, 172 | Station position (degrees, metres). |
| `station.file` | – | A station file with `station_lat`, `station_lon`, `altitude_m` (optionally `tz`). If neither `file` nor `lat`/`lon` is given, `samples/station.yaml` is read when it exists; otherwise the rounded default position above is used. Set the `station:` block explicitly for reproducible scenarios. |
| `station.horizon_deg` | `0` | Elevation at which a satellite counts as visible. |
| `station.fade_s` | `15` | Satellite signals rise from pure noise to their link-budget level over this long after AOS, and fall back to noise over this long before LOS (raised-cosine amplitude). There is no terrain/horizon map. A pass shorter than twice this never reaches full level. `0` switches it off. |
| `station.tz` | `+03:00` | Station local time zone, used only for file names. |
| `band` | `vhf137` | Band directory under `src/assist/bands/`: front-end defaults, calibrations, named channels. |
| `profile` | – | Directory with measured data (`profiles/vhf137`): noise shape and ADC noise level, METEOR and ORBCOMM calibrations. Without it the band's flat-shape defaults apply. |
| `frontend` | from the band | Receiver model, see below. |
| `antenna.pattern` | `qfh_poly` | Receive antenna gain versus elevation, see below. |
| `elements.files` | `[]` | Element-set files (OMM CSV or JSON, or TLE) for catalog orbits. |
| `elements.latest` | `false` | Use today's (UTC) `active_<date>.csv` and `satnogs_<date>.csv` from `elements/`; those missing are downloaded first, as `assist fetch_omm` does (network only then; if the download fails the newest existing files are used, with a warning). |
| `elements.latest_existing` | `false` | Use the newest `active_*` and `satnogs_*` files already in `elements/`; no network. Exclusive with `latest`. Either flag adds its files before `elements.files`. |
| `include` | – | A path or list of paths of YAML files merged underneath this one. |
| `sinks` | `[]` | Output settings: `rtl_tcp` and `fft` entries. |
| `sources` | `[]` | The signal sources. |

**`frontend`** is merged over the band's defaults, so you only give what you change:

| Field | Band default | Meaning |
|---|---|---|
| `fc_hz`, `fs_hz` | 137.0e6, 2.4e6 | Centre frequency and sample rate (a multiple of 100 Hz). Leave them: the stream is meant to be 137.000 MHz at 2.4 MS/s. |
| `lna.gain_db`, `lna.nf_db` | 20, 1.0 | LNA gain and noise figure. |
| `tuner_nf_db` | 3.5 | Tuner noise figure. |
| `saw_loss_db` | 1.465 | SAW filter loss. |
| `t_ant_k` | 500 | Antenna noise temperature. Together with the three fields above it sets the system temperature `T_sys`, to which every C/N0 level refers. |
| `noise_sigma_counts` | profile's (26.7) | ADC noise sigma per I/Q component, in counts. Default: the profile's, else the band's. |
| `tuner_gain_db`, `image_db`, `iip3_dbm` | 49.6, −75, −30 | Accepted and recorded, but not modelled yet (no image, intermodulation or gain-dependent effects). |
| `ppm` | `0` | Crystal error. Only `0` is supported. |

**`antenna.pattern`** (receive gain; each emitter has its own `pattern` for its transmit gain, same forms):
`qfh_poly` (a quadrifilar-helix polynomial in elevation, the default), `isotropic`, or a mapping:
`{type: poly, coeffs_db: [...], max_deg: 90}`, `{type: cardioid, n: 1.5, peak_dbi: 2}`, `{type: isotropic, gain_dbi: 0}`.

### Sources

A **source** is one transmitting object (or one source of interference) with one or more emitters.

| Field | Default | Meaning |
|---|---|---|
| `id` | required | Unique name. Used by `--set`, by merging, and to key the random draws. |
| `preset` | – | Start from a [preset](#presets). Your keys override the preset's. |
| `platform` | `satellite` | `satellite`: has an `orbit`, Doppler, a link budget and passes. `ground`: a terrestrial transmitter (aircraft, ground station) with a fixed received level and no geometry. `local`: interference made inside the receiver (birdies, USB noise), a fixed level and no antenna. |
| `name` | – | Free label. |
| `orbit` | – | Required for satellites, forbidden otherwise. See [Orbits](#orbits). |
| `in_catalog` | `true` for catalog orbits, `false` for synthetic ones | Whether `assist catalog` lists the object. |
| `catalog_norad` | – | NORAD id under which a synthetic object appears in the catalog (needed with `in_catalog: true`; must not collide with a real one). |
| `channels` | – | A list of frequencies (Hz) or channel names (`S-5`) of this source. An emitter's `channel: 0` picks item 0. Mostly used with a random expression, as in the `orbcomm` preset. |
| `emitters` | required | List of [emitters](#emitters) (at least one; supplied by the preset if you use one). |

### Orbits

Exactly one of three forms:

```yaml
# 1. A catalog object: its element set is looked up in elements.files (the set nearest the clock start, or `epoch`).
orbit:
  norad: 59051                  # required
  epoch: "2026-11-02T04:39:05Z" # optional: choose the element set nearest this time
  perturb:                      # optional: make the TRUE orbit differ from the published one
    along_track_s: 40           #   > 0: the satellite is this many seconds behind the published position
    raan_deg: 0.05              #   added to RAAN
    inc_deg: 0.0                #   added to the inclination
    dn_rev_day: 0.0             #   added to the mean motion (error grows with time from the epoch)

# 2. A synthetic pass: a circular orbit solved so that the satellite rises at `aos` and culminates at max_el_deg.
orbit:
  synthetic_pass:
    aos: +2min                  # required: ISO time, or an offset from the clock start (+2min, -4min)
    max_el_deg: 45              # required, 0 < el <= 90
    direction: desc             # asc | desc (default: drawn per source, keyed)
    side: east                  # east | west: the sky half of the culmination (default: drawn)
    alt_km: 820                 # default 820
    inc_deg: 98.7               # default 98.7

# 3. Kepler elements, as SGP4 mean elements:
orbit:
  kepler:
    epoch: "2026-11-03T00:00:00Z"   # required
    alt_km: 780                     # exactly one of alt_km and a_km
    i_deg: 82.5                     # required
    e: 0.0
    raan_deg: 40
    argp_deg: 0.0
    m_deg: 90
    bstar: 0.0
```

With `perturb`, the truth uses the perturbed orbit while `assist catalog` still reports the published one, mimicking
the error of real element sets. Catalog sources get their `name` and NORAD id in the truth. Element sets older than 14 days
at the clock start produce a warning. Orbits are propagated with SGP4.

### Emitters

An emitter is one transmitted signal of a source.

| Field | Default | Meaning |
|---|---|---|
| `id` | required | Unique within the source. |
| `enabled` | `true` | `false` removes it. |
| `signal` | required | `{kind: ..., params: {...}}`, see [Signal kinds](#signal-kinds-and-their-parameters). |
| `freq_hz` | – | Nominal rest frequency in Hz. Event signals may give a **list**: each event is on one of them (drawn per event). Exactly one of `freq_hz` and `channel` is required. |
| `channel` | – | An index into the source's `channels`, or a channel name from the band (`S-5`, `orbcomm_gateway`). |
| `offset_hz` | `0` | Fixed offset added to the frequency (rest-frequency error). |
| `oscillator` | – | Satellites: `{pass_offset_sigma_hz: 50, drift_hz_s: 0.2}`. A random frequency offset (normal, sigma given) is drawn per pass, plus a linear drift within the pass; the sign of the drift is drawn per pass. |
| `schedule` | `continuous` | When the emitter transmits, see [Schedules](#schedules). |
| `am` | – | Amplitude modulation of the carrier by a tumbling spacecraft: `{period_s, depth_db, phase_rad}`. `phase_rad` defaults to a keyed draw. `null` (or omitting it) means a steady amplitude. |
| `level` | required | See [Levels](#levels). Not allowed for the wideband `noise_comb`, whose level is relative to the noise. |
| `per_event` | – | Per-burst frequency offset and level, see [Per-event draws](#per-event-draws). |
| `pattern` | `qfh_poly` | Transmit antenna pattern (satellites), same forms as `antenna.pattern`. |

Signals are normalised to unit mean power while on; the level sets the power. Frequencies are rounded to 1 mHz.

### Signal kinds and their parameters

`signal: {kind: <name>, params: {...}}`. Parameters not given take the defaults below; unknown parameters are errors.
Ranges (`[lo, hi]`) are inclusive and are drawn per burst. Kinds are *continuous* (always renderable, gated by the
schedule), *event* (the schedule places events and the signal decides their length and content) or *wideband*.

| Kind | Mode | Models | Parameters (default) |
|---|---|---|---|
| `cw` | continuous | Unmodulated carrier: zombies, beacons, birdies, the 138 MHz carrier. | none |
| `oqpsk` | continuous | METEOR-M LRPT: OQPSK with a root-raised-cosine pulse. | `rs` 72000 sym/s, `alpha` 0.6, `sps` 4, `span` 10, `framing` `meteor_cadu` (CADUs with the 0x1ACFFC1D marker, randomiser and K=7 r=1/2 convolutional code; payload random) or `random` |
| `oqpsk_burst` | continuous | ORBCOMM gateway downlink: bursts of OQPSK, 4–5 blocks every 0.95 s. | `rs` 28800, `alpha` 0.6, `sps` 4, `span` 10, `blocks` [4, 5], `block_ms` 62.5, `gap_ms` 2.5, `period_s` 0.95, `ramp_ms` 0.5 |
| `sdpsk` | continuous | ORBCOMM subscriber downlink: symmetric differential PSK. | `rs` 4800, `alpha` 0.4, `sps` 8, `span` 10 |
| `noise_band` | continuous | A band of noise with a Gaussian spectrum (wide birdies). | `width_hz` (**required**, full width at half maximum), `shape` `gaussian` |
| `lora` | event | SpaceBEE-style LoRa bursts, 125 kHz. Spreading factor and payload length are drawn per burst; airtime follows Semtech's formula. | `bw` 125000, `sf` [7, 12] (6..12), `cr` 1 (1..4), `preamble` 8, `payload_bytes` [10, 60] (0..255), `crc` true, `ldro` `auto`, `sync_word` 0x12 |
| `d8psk` | event | VDL Mode 2 bursts: differential 8-PSK, raised cosine, 16-symbol sync. | `rs` 10500, `alpha` 0.6, `sps` 4, `span` 8, `ramp_up` 5, `header` 17, `ramp_down` 3, `burst_ms` [20.0, 23.0] |
| `gfsk` | event | VDL Mode 4 style bursts: 19.2 kbit/s Gaussian FSK. | `rb` 19200, `h` 0.25, `bt` 0.28, `sps` 4, `span` 4, `ramp` 4, `transition_p` 0.5, `burst_ms` [20.5, 22.0] |
| `am` | event | Airband AM voice, push-to-talk: DSB with full carrier, synthetic syllables (no words), 20 ms PTT ramps. The level is the carrier's. | `mod_index` 0.8, `ptt_ramp_ms` 20, `audio_lo_hz` 300, `audio_hi_hz` 3000 |
| `noise_comb` | wideband | USB noise humps: a comb of noise ripples across the whole band that wanders in frequency. Added inside the receiver before the ADC. | `spacing_hz` 80000, 141176 or 150000 (a number, or a list to choose from per episode), `amplitude_db` [0.1, 0.3] (peak ripple), `wander_hz_s` 1500, `knot_s` 20, `ramp_s` 2 |

Rules that follow from the modes:

- **Event signals** (`lora`, `d8psk`, `gfsk`, `am`) need an event schedule (`poisson_events` or `exp_gaps`). Their burst
  length comes from the signal itself (`burst_ms`, LoRa airtime); `am` and `noise_comb` take it from the schedule's `duration`.
- **Continuous signals** (`cw`, `oqpsk`, `oqpsk_burst`, `sdpsk`, `noise_band`) accept any schedule; an event schedule
  gates them, with `ramp_ms` raised-cosine ramps centred on each event's edges.
- **`noise_comb`** (wideband) needs `platform: local`, an event schedule, and no `level`, `per_event` or `am`. Its
  episodes are the schedule's events.
- The `platform: local` kinds (`cw`, `noise_band`, `noise_comb`) are added inside the receiver: no antenna, no band-pass response.

### Schedules

`schedule: {type: ..., ...}`. The `type` selects the form; the default is `continuous`. Everything is a function of
absolute time, so any window can be rendered on its own.

| Type | Fields (default) | Behaviour |
|---|---|---|
| `continuous` | – | Always on. |
| `on_off` | `on_s` 40, `off_s` 180, `phase_s` (keyed draw), `ramp_ms` 5 | On for `on_s` out of every `on_s + off_s`, an `on` interval starting at `phase_s` (mod the period). Ramps are raised-cosine, centred on the edges. |
| `exp_gaps` | `mean_gap_s` 180, `min_gap_s` 20, `duration`, `ramp_ms` 0 | Bursts separated by `min_gap_s` plus an exponential gap (overall mean `mean_gap_s`), a fresh chain every UTC hour. Requires `0 <= min_gap_s <= mean_gap_s`, and **an event's longest possible duration must not exceed `min_gap_s`** (a violation is reported when the scene is built, e.g. by `passes`). |
| `poisson_events` | `rate_per_hour` (**required**), `duration`, `min_gap_s` 0, `ramp_ms` 0 | A Poisson number of events per UTC hour, starts uniform in the hour. `rate_per_hour` is a number or **24 values, one per UTC hour**. With `min_gap_s` > 0 the emitter is one transmitter: an event starting less than `min_gap_s` after the previous kept one's end is dropped, so bursts never overlap. |

`duration` (event schedules; used where the signal does not set its own length):
`{dist: fixed, s: 1.0}`, `{dist: lognormal, median_s: 1.0, sigma: 0.5}` or `{dist: exponential, mean_s: 1.0}`, all with a
cap `max_s` (default 3600).

Satellite emitters are additionally gated by visibility: nothing is transmitted when the satellite is below the horizon, and
the signal fades in and out over `station.fade_s` at the pass edges.

### Levels

Exactly one key, depending on the platform. All C/N0 values are in dB-Hz *against k·T_sys*, i.e. the SNR a
1 Hz-wide measurement at the emitter's frequency would read.

**Satellites:**

| Key | Meaning |
|---|---|
| `cn0_zenith_dbhz` | C/N0 of an overhead pass. The transmit power is solved from it; the signal then follows the pass geometry (range, elevation, both antenna patterns, Doppler). |
| `cal` | A calibrated level taken from the profile or band: `meteor_auto` (the calibration of METEOR-M2 3 or M2-4 for those catalog objects, else the joint one), `meteor_m2_3`, `meteor_m2_4`, `meteor`, `orbcomm`. |
| `p_tx_dbw` | Power into the transmit antenna. |

**Ground and local** (a fixed received level; no geometry): `cn0_dbhz` (received C/N0), or `rx_dbm` (power at the
antenna terminals).

`extra_loss_db` (default 0) can be added to any form: a loss in dB applied to the received level, e.g. `level: {cn0_dbhz: 48, extra_loss_db: 3}`.

### Per-event draws

For transmitters without geometry (aircraft), every burst can get its own frequency offset and level, drawn from the
schedule's keyed stream. `per_event` needs an event schedule.

```yaml
per_event:
  offset_hz: [-2000, 2000]                     # a number, or an inclusive [lo, hi] (uniform)
  level_db: {dist: normal, sigma_db: 5, min_db: -12, max_db: 12}   # relative to the emitter's level
  # or {dist: uniform, min_db: -10, max_db: 5}
```

`normal` is `N(0, sigma_db)` clipped to `[min_db, max_db]`. The truth's event list records each burst's offset
frequency and received level.

### Presets

A preset is a ready-made source; give it an `id` and fill what it leaves open. They live in `src/assist/presets/`.

| Preset | Platform | What it is | You must give |
|---|---|---|---|
| `meteor_lrpt` | satellite | METEOR LRPT on 137.9 MHz, continuous, `cal: meteor_auto` | `orbit` |
| `orbcomm` | satellite | Emitters `stx1`, `stx2` (SDPSK on two of S-1…S-12, drawn per source) and `gateway` (OQPSK bursts at 137.56 MHz; `enabled` is a coin flip by default). Level `cal: orbcomm`. | `orbit`; optionally `channels: [S-5, S-8]`, `gateway: {enabled: true}` |
| `spacebee` | satellite | LoRa bursts on 137.099 MHz or 137.9265 MHz (per burst), `exp_gaps` mean 180 s, `cn0_zenith_dbhz: 50` | `orbit`; optionally one `freq_hz` |
| `zombie` | satellite | A dead satellite: carrier keyed 40 s on / 180 s off, tumbling AM (7.5 s, 6 dB), oscillator offset and drift, `cn0_zenith_dbhz: 45` | `orbit`, `freq_hz` |
| `vdl_ground` | ground | VDL Mode 2 ground station at a fixed level, Poisson bursts with a dead time, 136.975 MHz | `level`, `schedule.rate_per_hour` |
| `vdl_aircraft` | ground | VDL aircraft: every burst its own offset (±2 kHz) and level (σ 5 dB) | `schedule.rate_per_hour`, `level` (the median) |
| `atc_am` | ground | Airband AM voice: Poisson push-to-talk transmissions, per-burst offset and level | `freq_hz`, `level`, `schedule.rate_per_hour` |
| `carrier_138` | ground | The unmodulated 138.1104 MHz carrier, Poisson bursts | `level`, `schedule.rate_per_hour` |
| `usb_humps` | local | USB noise humps, episodes of a wandering comb | – |
| `birdies` | local | Only the platform; list the tones as your own emitters | `emitters` |

The preset's emitter ids: `zombie` → `carrier`; `meteor_lrpt` → `lrpt`; `orbcomm` → `stx1`, `stx2`, `gateway`;
`spacebee` → `lora`; `vdl_ground`/`vdl_aircraft` → `vdl`; `atc_am` → `voice`; `carrier_138` → `carrier`;
`usb_humps` → `comb`.

How a source's keys combine with its preset:

- Keys of the **source** (`orbit`, `channels`, `platform`, ...) override the preset's.
- A key that **names an emitter** goes to that emitter: `gateway: {enabled: true}`, `comb: {enabled: false}`.
- An **emitter field** given on the source (`freq_hz`, `schedule`, `am`, `level`, `signal`, `offset_hz`, `pattern`, `per_event`,
  `oscillator`, `enabled`) is applied to **every** emitter of the preset. Single-emitter presets make this convenient:
  `{id: z1, preset: zombie, freq_hz: 136.5e6, schedule: {type: continuous}, am: null}`.
- Mappings merge key by key (`schedule: {on_s: 80, off_s: 80}` keeps the preset's `ramp_ms`). Changing a mapping's `type`/`kind`/`dist`
  replaces it, and choosing one of a set of alternatives drops the others (`level: {cal: ...}` replaces the preset's
  `cn0_zenith_dbhz`; `orbit: {kepler: ...}` replaces a `norad`).
- Or skip presets and write `emitters:` out in full (see `scenarios/signal_tour.yaml`).

### Includes, merging and overrides

**`include:`** merges other YAML files underneath the scenario. `bands/vhf137/terrestrial_default.yaml` is the station's measured
terrestrial environment (about 20 sources: VDL ground stations and aircraft on 136.925 and 136.975 MHz, 14 airband AM channels, the 138.1104
MHz carrier, USB humps and 15 birdies, with hourly rates and levels taken from the recording). Include paths are looked up relative to the
including file, then in the package.

**Merging**: mappings merge recursively; lists of mappings that all have an `id` merge by `id` (an entry with a new id is appended); any other list
is replaced. So you can adjust an included source by repeating its id:

```yaml
include: [bands/vhf137/terrestrial_default.yaml]
sources:
  - {id: carrier-138, carrier: {enabled: false}}        # switch off the 138.1104 MHz carrier
  - {id: usb-humps, comb: {schedule: {rate_per_hour: 6}}}  # more humps (the emitter is `comb`)
  - id: am-136.2000
    schedule: {rate_per_hour: 60}                       # busier ATC on 136.2 MHz
```

**`--set path=value`** overrides one field on the command line. The value is parsed as YAML (`137.35e6`, `true`, `[S-5, S-8]`); list
items are addressed by `id` (or index); the override is applied before presets are expanded, so it can reach anything a source key can.
Quote it in the shell:

```sh
uv run assist render scenarios/signal_tour.yaml -o out/x --format cu8 \
  --set seed=2 --set 'sources[sat-cw].freq_hz=136.6e6' --set 'sources[voice].voice.enabled=false'
```

### Random expressions

Any value can be a string with a random expression, drawn **once, at resolve time**, from the scenario seed and the field's path:

| Expression | Result |
|---|---|
| `uniform(a, b)` | a float in [a, b] |
| `normal(mu, sigma)` | a normal draw |
| `randint(a, b)` | an integer in [a, b] (inclusive) |
| `choice([x, y, z])` | one item |
| `sample([x, y, z], k)` | `k` distinct items, as a list |
| `bernoulli(p)` | `true` with probability `p` |

```yaml
freq_hz: "uniform(136.1e6, 136.9e6)"
enabled: "bernoulli(0.5)"
channels: "sample([S-1, S-2, S-3, S-4, S-5, S-6, S-7, S-8, S-9, S-10, S-11, S-12], 2)"
```

The draw is keyed by the source id and the field path, not by position, so it is fixed for a seed and unaffected by
other sources. Change `seed` (or the source id) for a different draw. `assist resolve` prints what was drawn.

### Sinks

`sinks:` is a list of one-key mappings.

```yaml
sinks:
  - rtl_tcp: {host: 127.0.0.1, port: 1234}     # used by `serve`; the command-line options win
  - fft: {width: 16384, row_seconds: 1, duration: 1h, min_db: 20, max_db: 50}    # used by `fft`
```

`fft` (a waterfall PNG; the image is `width` columns by `duration / row_seconds` rows, oldest at the top):

| Key | Default | Meaning |
|---|---|---|
| `width` | 16384 | FFT size, a power of two from 64 to 2^20. Power is scaled so that a dB means the same for any width. |
| `row_seconds` | 1 | Seconds per row (a row is the average of whole FFTs, so it is accurate to within one FFT). |
| `duration` | – | Stream length to plot; required here or as `--duration`. |
| `min_db`, `max_db` | automatic | Colour scale, both or neither. Automatic: from the mean level of the first 5 seconds. |

### Bands and profiles

- A **band** (`src/assist/bands/<name>/band.yaml`) holds the front-end `defaults`, the receiver's default noise level, the `calibrations` (link
  constants of METEOR-M2 3, M2-4 and ORBCOMM) and the named `channels`: ORBCOMM subscriber `S-1`…`S-12` (137.200–137.800 MHz) and `orbcomm_gateway`
  (137.560 MHz). Bands can also carry include files such as `terrestrial_default.yaml`.
- A **profile** (`profiles/vhf137/`) is measured data from a real recording: the noise power spectrum and ADC noise level (the
  noise floor of the stream), the METEOR and ORBCOMM calibrations (which supersede the band's), the birdie list, and the VDL/AM/carrier statistics
  behind `terrestrial_default.yaml`. Put `profile: profiles/vhf137` in a scenario to get the measured noise floor.
- To make a profile from your own recording, `assist profile build` (and `labels`, `report`, `terrestrial`) read a `baseband_*.wav` capture, waterfall `.raw`
  captures, a station file and element sets from `samples/` (`--samples DIR`), and write the profile directory. `assist profile terrestrial --band` regenerates
  `terrestrial_default.yaml` from it. The finished profile in the repository is all a scenario needs.

## The example scenarios

All are in `scenarios/` and validate on a fresh checkout (they read `elements/` and `profiles/`).

| File | Shows |
|---|---|
| `realtime.yaml` | **The live station.** Starts `now`, serves over rtl_tcp. Noise floor; birdies and USB humps; VDL (ground stations and aircraft) and ATC AM; ten Starlinks carrying LoRa on 137.9265 MHz; METEOR-M2 3 and M2-4; the eight ORBCOMM FM1xx satellites of `active.csv` with random channels and the gateway on for three; five zombies on hand-picked frequencies (TRANSIT 5B-5 and ALOUETTE 2 continuous, OPS 7613 ×2 and OPS 7898 80 s on / 80 s off, none tumbling). |
| `signal_tour.yaml` | A tour of the options with no external data: every schedule type and platform, `per_event`, hourly rate lists, absolute levels, random frequencies, a `frontend` and `antenna` override, an `fft` sink. |
| `catalog_errors.yaml` | The relation between true orbit and catalog: a perfect entry, a perturbed one, one left out, a synthetic one listed under NORAD 99001, an unlisted synthetic one and a Kepler orbit. For `assist catalog`. |
| `meteor_lora.yaml` | Both METEORs back to back with ten Starlink-borne LoRa emitters on the overlapping channel (fixed date). |
| `realistic.yaml` | A mixed band: a low METEOR-M2 4 pass with LoRa on top, ORBCOMM without gateway, a 60 s / 10 s zombie, VDL, ATC, birdies, humps, written out explicitly (fixed date). |
| `terrestrial_hour.yaml` | The terrestrial environment for the hour of the station's recording, plus ORBCOMM FM117 and METEOR. |
| `serve_demo.yaml`, `serve_satellites.yaml`, `serve_terrestrial.yaml` | Live (`now`) scenarios with synthetic passes placed around the launch time: zombies; METEOR + ORBCOMM + LoRa; the terrestrial environment alone. |
| `zombie_single.yaml` | The smallest useful scenario: one zombie on a 45° pass. |
| `example.yaml` | Catalog orbits (both METEORs, TRANSIT 5B-5) beside a synthetic zombie. |

## Recipes

**A quiet band with one zombie** (see `zombie_single.yaml`):

```yaml
version: 1
seed: 1
clock: {start: "2026-11-04T12:00:00Z", duration: 18min}
profile: profiles/vhf137
sources:
  - id: z1
    preset: zombie
    orbit: {synthetic_pass: {aos: +1min, max_el_deg: 45, direction: desc, side: east}}
    freq_hz: 137.35e6
```

**A real satellite as a zombie that is on 80 s and off 80 s, not tumbling:**

```yaml
elements: {files: [elements/satnogs_2026-11-02.csv]}
sources:
  - id: ops7898
    preset: zombie
    orbit: {norad: 5680}
    freq_hz: 136.5213e6
    schedule: {on_s: 80, off_s: 80}
    am: null
```

**ORBCOMM with random channels and the gateway on:**

```yaml
  - {id: fm108, preset: orbcomm, orbit: {norad: 41187}, gateway: {enabled: true}}   # channels drawn from the seed
  - {id: fm117, preset: orbcomm, orbit: {norad: 41188}, channels: [S-5, S-8], gateway: {enabled: false}}
```

**A ground VDL station and aircraft on one frequency:**

```yaml
  - {id: vdl-gs, preset: vdl_ground, freq_hz: 136.975e6, level: {cn0_dbhz: 70}, schedule: {rate_per_hour: 600}}
  - {id: vdl-ac, preset: vdl_aircraft, freq_hz: 136.975e6, level: {cn0_dbhz: 61}, schedule: {rate_per_hour: 1200}}
```

**A birdie and a wide spur** (receiver spurs are `platform: local`):

```yaml
  - id: spurs
    platform: local
    emitters:
      - {id: tone, signal: {kind: cw}, freq_hz: 137.1355e6, level: {cn0_dbhz: 21}}
      - {id: wide, signal: {kind: noise_band, params: {width_hz: 2000}}, freq_hz: 136.0502e6, level: {cn0_dbhz: 32}}
```

**A satellite whose catalog entry is wrong, for testing a tracker:** see `catalog_errors.yaml`; then
`assist catalog scenarios/catalog_errors.yaml -o out/catalog.tle` is what the tracker is given and
`assist truth ...` is what really happened.

**Fresh element sets:** `elements: {latest: true}` in the scenario downloads today's snapshots when needed (or `uv run assist fetch_omm`, then
`latest_existing: true` or `elements.files`). Passes of real satellites are only where the real ones are if the element sets are recent.

## Outputs

**`truth`** writes `truth.csv` (one row per emitter per `--step`) and `truth.events.csv` (one row per burst). Truth rows have:
`t_utc, source_id, emitter_id, signal, norad, in_catalog, f_rest_hz, f_tx_hz, f_rx_hz, doppler_hz, el_deg, az_deg, range_km,
range_rate_mps, rx_dbm, cn0_dbhz, active_fraction`. Event rows: `start, end, source_id, emitter_id, signal, f_rx_hz, rx_dbm, cn0_dbhz, params`
(for event signals: spreading factor, payload length, per-event offset, ...). `--format jsonl` writes JSON lines, `--platforms`
filters by `satellite`, `ground`, `local`.

**`render`** and **`serve`** write `run.json` next to their output: the resolved scenario and its hash, the package version and engine,
the window as absolute sample indices, and per-file first-sample times. `run.json` is provenance, not truth; truth is always
recomputed from the scenario. The scenario hash is the sha256 of the canonical resolved JSON.

**`fft`** writes `waterfall_137000000Hz_HH-MM-SS_DD-MM-YYYY.png` and `run.json`.

Default output directories (`out/`, `temp/`) are gitignored.

## Element sets

Catalog orbits need element sets in an OMM CSV (retlector.eu layout, with the header `OBJECT_NAME,OBJECT_ID,EPOCH,MEAN_MOTION,...`), OMM JSON or TLE. The
snapshots of 2026-11-02 (not tracked by git; run `assist fetch_omm` to get current ones) are in `elements/`: `active_2026-11-02.csv` (active satellites) and `satnogs_2026-11-02.csv` (the SatNOGS feed,
including TRANSIT 5B-5, ALOUETTE 2 and the OPS objects). A scenario takes, per object, the element set whose epoch is nearest the clock start, and
`assist` warns when it is more than 14 days older than the window. METEOR-M2 3 (57166), METEOR-M2 4 (59051), the ORBCOMM FM satellites and the Starlinks
are in `active.csv`.

## Development

```sh
uv sync --all-groups        # installs everything and builds the C++ extension assist._dsp
uv run pre-commit install   # ruff, ruff-format, ty on commit
uv run pytest
```

- **C++ edits** under `cpp/` need no reinstall: the editable install rebuilds `assist._dsp` on the next import (a few seconds).
- **C++ kernels** (`cpp/`): overlap-add synthesis, Philox4x32-10 + Box–Muller noise shaped by overlap-save, the ADC, payload bytes, pulse shaping, LoRa chirps,
  the carrier oscillator, and METEOR framing (convolutional encoder, CCSDS randomiser). pocketfft is vendored in `cpp/third_party/`. Each kernel has a numpy
  twin in `src/assist/dsp_ref/`, compared in `tests/test_dsp_ref.py`.
- **`ASSIST_NATIVE=ON uv sync --reinstall-package assist`** builds with `-march=native` (local use only); a plain `uv sync --reinstall-package assist` switches back.
- **Speed**: one core renders the realtime scenario at ~7x real time; the file render runs in parallel worker processes (20–30x on 6 cores) and gives the same bytes for any worker count.
- Large local data (recordings, waterfall captures) lives in `samples/`, which is gitignored; CI tests never need it, and the local-data tests skip when it is absent.
