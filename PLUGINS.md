# Writing a plugin

ASSIST has three extension points: **signals** (how an emitter is modulated), **platforms** (where a source is and
how it moves) and **sinks** (where the IQ stream goes). The modules shipped with ASSIST and the ones you write use
the same three decorators. A plugin is a Python file, or an installed package, that a scenario names under
`plugins:`. No file of ASSIST itself has to change.

This guide covers how a scenario loads a plugin, the rules for random numbers, and how to write each of the three
kinds, with a working example for each. [README.md](README.md) describes scenarios and commands;
[HOW-IT-WORKS.md](HOW-IT-WORKS.md) describes the internals.

## Contents

- [The shortest route](#the-shortest-route)
- [How a scenario loads plugins](#how-a-scenario-loads-plugins)
- [Trust](#trust)
- [Randomness](#randomness)
- [Writing a signal](#writing-a-signal)
- [Writing a platform](#writing-a-platform)
- [Writing a sink](#writing-a-sink)
- [Testing a plugin](#testing-a-plugin)
- [Packaging](#packaging)
- [The API version](#the-api-version)
- [What is in `assist.api`](#what-is-in-assistapi)

## The shortest route

1. Copy the example nearest to what you need:

   | Example | Kind | What it shows |
   |---|---|---|
   | [`examples/plugins/pretty_gfsk.py`](examples/plugins/pretty_gfsk.py) | signal | Bursts placed by the schedule (an *event* signal), per-burst draws, counter-based data bits |
   | [`examples/plugins/jitter_platform.py`](examples/plugins/jitter_platform.py) | platform | Geometry as a function of time, Doppler from a changing range, per-source draws |
   | [`examples/plugins/tcp_sink.py`](examples/plugins/tcp_sink.py) | sink | A TCP server built on the shared serving code |

2. Import only `assist.api`, declare `ASSIST_API = "1.0"`, and register your class with a decorator.
3. Name the file in a scenario and look at what is registered:

   ```yaml
   plugins:
     - path: my_signal.py
   ```

   ```sh
   uv run assist plugins list scenario.yaml
   uv run assist run scenario.yaml
   ```

4. Run the conformance check for your kind of module (see [Testing a plugin](#testing-a-plugin)).

## How a scenario loads plugins

A scenario lists every external module it uses. Only the listed modules are imported. An installed plugin package
that no scenario names is never loaded, so a scenario gives the same result on any machine that has the modules it
lists.

```yaml
plugins:
  - path: plugins/msk.py        # a file
  - psk8                        # a name: an entry point of the group `assist.plugins`, else an importable module
  - module: mypkg.modems        # an importable module
    only: [psk8]                # use only these of the names the module registers
    prefix: lab                 # register them as lab.<name>
```

| Form | Meaning |
|---|---|
| `path:` | A Python file. A relative path is looked up in the working directory first, then in the directory of the scenario file. |
| a bare string, or `name:` | The entry point of that name in the group `assist.plugins`. If there is none, an importable module of that name. |
| `module:` | An importable module (dotted name). |
| `only:` | Optional. A list of registered names to use; the module's other names are left out. |
| `prefix:` | Optional. Registers the module's names as `prefix.name`. |

Rules:

- **Names.** A plugin registers plain names (`msk`), like the core modules. A name that is already a core name, or
  that two listed plugins both register, is an error that names both. Give one entry a `prefix:` (or an `only:`
  list) to resolve it.
- **Errors.** A file that is missing, a module that fails to import, a module that registers nothing, and an entry
  listed twice are errors that name the entry.
- **Undeclared kinds.** A scenario that uses `kind: msk` without listing the plugin gets an error that lists the known
  names and says the module must be in `plugins:`.
- **`--plugin P`** on the command line adds one more plugin for that run (a `.py` path or a name). It is for
  experiments. A scenario that is meant to be kept should list its plugins itself.

**Provenance.** When a scenario is resolved, each `plugins:` entry gets the plugin's `api` (the version it declares),
its `version` (the module's `__version__`, else the version of its installed distribution) and, for a file, the
`sha256` of the file. These are part of the resolved scenario, so they are part of the **scenario hash**: a changed
plugin is a changed scenario. A resolved scenario refuses a plugin file whose content has changed since it was
resolved. `run.json` records for each loaded plugin its label, version, API version, sha256, module, file and
distribution, and the names it registered. A scenario without plugins has no `plugins` key in its canonical form,
so its hash does not depend on any of this.

## Trust

A plugin is ordinary Python. It runs in the same process as ASSIST (and in the render worker processes) with the
full rights of the user who started it. ASSIST does not sandbox or restrict it. List a plugin in a scenario only if
you would run that file with `python` yourself. Because nothing is loaded unless a scenario names it, installing a
plugin package does not by itself run any of its code.

## Randomness

A scenario has one `seed`. Every random number in a render comes from it, through keys made of stable names. This
is what lets any window of a scenario be rendered on its own, in any order and on any number of worker processes,
with the same bytes. A plugin must keep to the same rules.

**A plugin never sees the seed.** It gets an `RngScope`: a part of the key space under a fixed prefix.

| Module | Where its scope is | Prefix of every key |
|---|---|---|
| signal | `self.keys` | `("signal", kind, source id, emitter id)` |
| platform | `ctx.scope` | `("platform", platform name, source id)` |
| sink | `ctx.scope` | `("sink", sink name)` |

The prefix holds the kind of module and its registered name. Two modules that both use the purpose `"data"`
therefore get different numbers, and adding or removing a source, an emitter or a plugin never changes the draws of
another one.

A scope gives two things:

- **`scope.key64(*parts)`** returns a 64-bit key for a **counter-based** generator. Use it for long sequences:
  data bits, noise. The kernel takes the key and a counter and returns the block of that counter directly:
  `dsp.random_bytes(key, block, n)` gives the `n` bytes of block `block`, and `dsp.white(key, h0, nhops, hop)` gives
  complex Gaussian noise for hops `h0 .. h0 + nhops - 1`. The counter must be an **absolute** index (a block
  number on an absolute grid, an event's start time in microseconds), never "the n-th call".
- **`scope.stream(*parts)`** returns a numpy `Generator`. Use it for **a few draws about one thing**: the parameters
  of one event, an offset of one pass, a property of one source. Put the absolute identity of that thing in `parts`
  (`scope.stream("pass", aos_key)`), and draw the values in a fixed order.

`scope.child(*parts)` gives a narrower scope with a longer prefix.

Key parts are strings or non-negative integers.

What a plugin must not do:

- call `np.random.default_rng()`, `random`, or use any global or time-dependent state for content;
- keep a generator and draw from it across `render` calls: the result would depend on the order and the size of
  the windows asked for;
- use a counter that is relative to the window (`n - n0`) instead of the absolute index.

**Counter-based rendering is the contract.** A signal must return the same samples for an absolute index whatever
window asks for it. The engine renders in 1 s chunks, in any order, in several processes, and an rtl_tcp client can
join at any time. `assist.testing.check_signal` checks this by rendering a span whole and in pieces.

## Writing a signal

A signal produces **complex64 baseband samples with unit mean power while it is on**, at a *narrow rate* that the
engine picks from what the signal asks for. The engine does everything else: carrier frequency, Doppler, level, the
schedule's on/off, fading, the antenna pattern, mixing to the station's sample rate, noise and the ADC. A signal is
numpy code; it needs no C++.

There are three modes (`mode` on the class):

| Mode | The signal implements | Core examples |
|---|---|---|
| `continuous` | `render(n0, n, rate)`: samples at any absolute index | `cw`, `oqpsk`, `sdpsk` |
| `event` | `draw_event`, `max_duration_s`, `render_events`: bursts placed by the emitter's schedule | `lora`, `d8psk`, `gfsk`, `am` |
| `wideband` | `render_wide`: interference inside the receiver, at the full sample rate, on a `local` source | `noise_comb` |

Every signal gives three numbers the engine uses to choose the narrow rate:

- `bandwidth_hz()`: the occupied bandwidth of the baseband signal;
- `preferred_rate()`: the lowest narrow rate it wants (for example samples per symbol times the symbol rate);
- `rate_step()`: the rate is a multiple of this (default 100). Return the symbol rate to get a whole number of
  samples per symbol.

The engine may pick a higher rate than `preferred_rate()` (a large Doppler needs room), always a multiple of
`rate_step()`. Compute the samples per symbol from the `rate` argument of `render`, not from the parameters.

Sample `n` at rate `rate` is at Unix time `n / rate`. Indices are large (about 10^14), so keep them in `int64`
and reduce them (`n % sps`, `n // sps`) before converting to float.

### A continuous signal

This is a complete plugin: binary FSK with modulation index 1. The data bits come from `dsp.random_bytes` in blocks
on an absolute bit grid, and the phase at the start of bit `k` is `pi * k`, so no history is needed.

```python
"""`fsk2`: continuous binary FSK with modulation index 1 (the two tones are rb apart)."""

from typing import Any, override

import numpy as np
from pydantic import Field

from assist import api

ASSIST_API = "1.0"

BLOCK = 4096  # bits per block of random data


class Fsk2Params(api.SignalParams):
    rb: int = Field(9600, gt=0, description="Bit rate (bits per second).")
    sps: int = Field(8, ge=4, description="Samples per bit at the preferred narrow rate.")


@api.register_signal("fsk2")
class Fsk2(api.Signal):
    Params = Fsk2Params
    mode = "continuous"

    def __init__(self, params: api.SignalParams, keys: api.Keys, dsp: Any) -> None:
        super().__init__(params, keys, dsp)
        assert isinstance(params, Fsk2Params)
        self.rb = params.rb
        self.sps = params.sps
        self.key = keys.key64("bits")  # one key for all data bits of this emitter

    @override
    def bandwidth_hz(self) -> float:
        return 3.0 * self.rb

    @override
    def preferred_rate(self) -> int:
        return self.sps * self.rb

    @override
    def rate_step(self) -> int:
        return self.rb  # the narrow rate is a whole number of samples per bit

    def bits(self, k0: int, k1: int) -> np.ndarray:
        """Bits k0 .. k1 - 1 on the absolute bit grid (bit k starts at Unix time k / rb)."""
        blocks = range(k0 // BLOCK, (k1 - 1) // BLOCK + 1)
        data = np.concatenate([api.unpack(self.dsp.random_bytes(self.key, b, BLOCK // 8)) for b in blocks])
        return data[k0 - blocks[0] * BLOCK : k1 - blocks[0] * BLOCK]

    @override
    def render(self, n0: int, n: int, rate: int) -> np.ndarray:
        sps = rate // self.rb
        idx = n0 + np.arange(n, dtype=np.int64)  # absolute narrow indices
        k = idx // sps  # the bit each sample is in
        sign = 2.0 * self.bits(int(k[0]), int(k[-1]) + 1)[k - k[0]] - 1.0
        # Each bit turns the phase by +-pi, so every bit starts at phase pi * k: no history is needed.
        phase = np.pi * ((k % 2) + sign * (idx % sps) / sps)
        return np.exp(1j * phase).astype(np.complex64)
```

A scenario uses it like a core kind:

```yaml
plugins:
  - path: fsk2.py
sources:
  - id: beacon
    platform: ground
    emitters:
      - id: data
        signal: {kind: fsk2, params: {rb: 9600}}
        freq_hz: 137.5e6
        level: {cn0_dbhz: 70}
```

Points to note:

- **`Params`** is a pydantic model derived from `api.SignalParams`. Unknown keys are errors. Validation fills in the
  defaults, so the resolved scenario lists every parameter and the scenario hash covers them.
- **The constructor** receives the validated parameters, the emitter's scope (`keys`) and the kernels (`dsp`).
  Take `key64` keys once there.
- **No state between calls.** A cache of something that depends only on absolute indices is fine (the examples
  cache filter taps and the last burst). A running phase, a generator or a buffer position is not.
- **Pulse shaping.** For a linear modulation with a root-raised-cosine pulse, derive from `api.ShapedSignal`
  (parameters from `api.ShapedParams`), as the core `sdpsk` does. `api.rrc_taps` gives the taps alone.

### An event signal

[`pretty_gfsk.py`](examples/plugins/pretty_gfsk.py) is the example. The emitter's `schedule` (`exp_gaps` or
`poisson_events`) decides when events start. The signal decides what an event is:

- `draw_event(p, g)` (a classmethod) returns `(duration in s, parameters)` for one event. `g` is the schedule's
  generator for that event's slot, so the draw is keyed and independent of the window. The parameters go into
  `Event.params` and into the truth event list. A duration of `None` lets the schedule's own duration distribution
  decide.
- `max_duration_s(p)` (a classmethod) bounds the duration, so the engine knows how far back to look for events
  that reach into a window.
- `render_events(events, n0, n, rate)` returns the samples `[n0, n0 + n)` with the given events in them and zero
  elsewhere. Each event must be computed from its own start, so its samples are the same in every window. The
  example forms the burst from `ev.start` and uses the start time in microseconds as the counter of its data bits.
- `event_bounds(ev)` returns the time span outside which the event's samples are exactly zero (default:
  `ev.start, ev.end`). Override it when the burst is aligned to a grid or has filter tails.

An `Event` has `start`, `end` (Unix seconds) and `params`.

### Where the signal's level comes from

The signal returns unit power. The emitter's `level` sets the received power: `cn0_dbhz` or `rx_dbm` on a platform
with a received level, or `p_tx_dbw` / `cn0_zenith_dbhz` / `cal` through the link budget on the core `satellite`.
A signal that is off for part of the time (bursts, ramps) has unit mean power over the time it is on.

## Writing a platform

A platform says where a source is at any absolute time and when it is visible. A source selects it with
`platform: name` and gives its settings in `platform_params:`.

[`jitter_platform.py`](examples/plugins/jitter_platform.py) registers `tethered`: a transmitter in a fixed
direction whose distance swings, so the carrier shows Doppler.

```yaml
plugins:
  - path: examples/plugins/jitter_platform.py
sources:
  - id: balloon
    platform: tethered
    platform_params: {az_deg: 200, el_deg: 35, range_m: 3000, swing_m: 50, period_ms: 5000}
    emitters:
      - id: beacon
        signal: {kind: cw}
        freq_hz: 137.3e6
        level: {cn0_dbhz: 50}
```

A platform class has:

- **`Params`**: a pydantic model derived from `api.PlatformParams`, validated from `platform_params:`.
- **`build(ctx)`** (a classmethod): makes the platform of one source from a `PlatformContext`: `source_id`, `name`,
  `params` (validated), `station`, `horizon_deg`, `scope` (its `RngScope`) and `source` (the resolved source).
- **`geometry(t_unix)`**: an `api.Geometry` (elevation, azimuth, range, range rate, off-nadir angle, as arrays) at
  the given absolute times. It must be a pure function of the time.
- **`passes(t0, t1)`**: the visibility windows that overlap `[t0, t1)`, as `api.Pass` objects. Each pass is returned
  whole, and its `aos_key` (an integer) must not depend on the window asked for: per-pass random draws are keyed by
  it. The example returns one pass per UTC day.
- **`max_speed_km_s()`**: an upper bound of the range rate. It sets how much room the emitter's Doppler gets when
  the narrow rate is chosen.
- **`geometry_ms(k_ms)`** (optional): geometry at integer Unix milliseconds, the grid the renderer uses. Override it
  to form the time from the integers when float seconds are not exact enough. The example does, so its swing phase
  is exact.

**Capabilities.** The rules about what a source on a platform may have are class attributes, not names. The schema,
the resolver and the scene builder read them.

| Attribute | Default | Meaning |
|---|---|---|
| `fixed` | `False` | No geometry: always visible, no Doppler. `geometry` and `passes` are not used. |
| `needs_orbit` | `False` | The source must have an `orbit`. Core `satellite` only. |
| `catalog` | `False` | The source may be in the element catalog. Core `satellite` only. |
| `supports_oscillator` | `False` | Emitters may have per-pass oscillator terms (offset and drift per pass). |
| `level_mode` | `"received"` | `"received"`: the level is `cn0_dbhz` / `rx_dbm`. `"link_budget"` is for the core `satellite` only. |
| `fades_at_horizon` | `False` | `station.fade_s` is applied after AOS and before LOS. |
| `allows_wideband` | `False` | Wideband signals may sit on it. Needs `fixed = True`. |

An external platform supplies its own geometry and takes a **received level**. Orbits, the catalog and the link
budget (range, antenna patterns, calibrations) stay in the core `satellite` platform; registering an external
platform with `needs_orbit`, `catalog` or `level_mode = "link_budget"` is refused. A fixed platform cannot have
oscillator terms or a fade.

`assist truth --platforms tethered` lists the direction, range and range rate the platform gives.

## Writing a sink

A sink receives the station's stream: interleaved unsigned 8-bit I and Q, as an RTL-SDR gives. A scenario lists
sinks under `sinks:`, each as a one-key mapping of the sink's name to its settings, and `assist run` drives them:

```yaml
sinks:
  - wav: {rf64: true}
  - power_log: {name: power.csv}
```

The runner builds the renderer once, checks that no file is overwritten (without `--force`), writes `run.json`
before and after, maps errors to exit codes, and hands the stream to the sinks. A sink does none of that. Settings
come only from the scenario (`--set 'sinks[power_log].name=p.csv'` overrides one). There are no sink-specific
command-line options.

There are two kinds:

- **Push sinks** (`api.Sink`) are given the stream piece by piece, in order, as fast as it renders. Several push
  sinks can share one run.
- **Serving sinks** (`api.ServingSink`) serve network clients and pull the stream at their own pace. A run has
  exactly one serving sink and no other sink. A scenario may list both kinds; then `--sink NAME` selects what runs.

A sink is built from its validated `Params` and a `SinkContext`: `fs`, `fc_hz`, `n0` (absolute index of the first
sample), `n` (number of samples, `None` for an endless served stream), `out` (the output directory), `tz`, `scope`
(its `RngScope`), `scenario_hash` and `wall_start`.

### A push sink

This complete plugin writes the mean power of every second of the stream to a CSV file.

```python
"""`power_log`: a CSV with the mean power of every second of the stream (dB relative to full scale)."""

from typing import override

import numpy as np
from pydantic import Field

from assist import api

ASSIST_API = "1.0"


class PowerLogParams(api.SinkParams):
    name: str = Field("power.csv", description="File name inside the output directory.")


@api.register_sink("power_log")
class PowerLog(api.Sink):
    Params = PowerLogParams

    def __init__(self, params: api.SinkParams, ctx: api.SinkContext) -> None:
        super().__init__(params, ctx)
        assert isinstance(params, PowerLogParams) and ctx.n is not None
        self.path = ctx.out / params.name
        self.pending = np.zeros(0, dtype=np.uint8)  # samples of a second that is not complete yet
        self.rows: list[str] = []

    @override
    def files(self) -> list[api.OutFile]:
        assert self.ctx.n is not None
        return [api.OutFile(self.path, 0, self.ctx.n, self.ctx.n0)]

    @override
    def write(self, u8: np.ndarray) -> None:
        # Pieces come in any length: work in whole seconds, so the file does not depend on how the stream is cut.
        data = np.concatenate([self.pending, u8])
        per_s = 2 * self.ctx.fs  # bytes per second (I and Q)
        whole = len(data) // per_s
        for sec in data[: whole * per_s].reshape(whole, per_s):
            x = (sec.astype(np.float32) - 127.5) / 127.5
            t = self.ctx.n0 // self.ctx.fs + len(self.rows)
            self.rows.append(f"{t},{10 * np.log10(2 * np.mean(x * x)):.3f}")
        self.pending = data[whole * per_s :]

    @override
    def close(self) -> None:
        self.path.write_text("unix_s,power_dbfs\n" + "".join(r + "\n" for r in self.rows))

    @override
    def summary(self) -> str | None:
        return f"power_log: {len(self.rows)} rows in {self.path.name}"
```

The methods of a push sink:

| Method | When | Purpose |
|---|---|---|
| `__init__(params, ctx)` | before the run | Validate and plan. Raise `ValueError` for settings that cannot work. |
| `planned()` | before the run | Samples the sink takes from the start of the stream (default: all). A sink that needs fewer finishes early. |
| `files()` | before and after | The files it writes, as `api.OutFile(path, first, n_samples, start_sample)`. Checked for overwriting, listed in `run.json`. |
| `open()` | once, before the first `write` | Open outputs. |
| `write(u8)` | for every piece | The next piece: a `uint8` array of interleaved I, Q, of any length. |
| `close()` | at the end | Finish the outputs. Raise if the sink did not get what it planned. |
| `abort()` | on an error or Ctrl-C | Keep whatever is usable. |
| `info()` | at the end | Facts for `run.json` (must be JSON data). |
| `summary()` | at the end | One line printed at the end of the run, or `None`. |

**The output must not depend on how the stream is cut into pieces.** The length of the pieces varies with the
number of worker processes and with the other sinks in the run. The example buffers until it has a whole second.
`assist.testing.check_sink` feeds the same stream in different cuts and compares the files.

### A serving sink

[`tcp_sink.py`](examples/plugins/tcp_sink.py) serves the stream on a plain TCP socket as cu8, cs16 or cf32. It is
short because the serving code is shared:

- `api.StreamSink` is a serving sink made of a server. Its `Params` derive from `api.ServingParams` (`host`, `port`,
  `pacing`, `lookahead`). The subclass implements `make_server(ahead)`; `self.server_args()` gives the keyword
  arguments every server takes.
- `api.StreamServer` accepts clients, paces the stream (`realtime` or `unpaced`), starts it live when the scenario
  clock is `now`, sends it to every client, and disconnects a client that falls too far behind. A subclass adds a
  protocol with three hooks: `greeting()` (bytes sent on connect), `read_client(...)` (what to do with what a client
  sends) and `encode(chunk)` (how a chunk goes on the wire).
- `api.RenderAhead` is the single render thread that fills a bounded ring ahead of the clients.

The core `rtl_tcp` sink is the same structure with the rtl_tcp header as its greeting and the rtl_tcp command
parser in `read_client`.

## Testing a plugin

`assist.testing` has one check per kind of module. Each raises `ConformanceError` (an `AssertionError`) that says
what is wrong, and returns nothing when the module conforms.

```python
from assist import testing


def test_my_signal() -> None:
    mod = testing.load_plugin("plugins/fsk2.py")
    testing.check_signal(mod.Fsk2)
    testing.check_signal("fsk2", {"rb": 4800, "sps": 5}, seed=3)


def test_my_platform() -> None:
    mod = testing.load_plugin("examples/plugins/jitter_platform.py")
    testing.check_platform(mod.Tethered, {"range_m": 3000, "swing_m": 50})


def test_my_sink() -> None:
    mod = testing.load_plugin("plugins/power_log.py")
    testing.check_sink(mod.PowerLog, seconds=3.0)
```

| Check | What it verifies |
|---|---|
| `check_signal(cls, params)` | Unknown parameters are refused. The narrow rate is consistent. A span rendered whole and in pieces (with fresh instances) gives identical samples. The output is complex64 with unit mean power while on. An event signal is exactly zero outside its events' bounds. |
| `check_platform(cls, params)` | Unknown parameters are refused. The same seed gives the same platform. `geometry` is a pure function of time and agrees with `geometry_ms`. `passes` do not depend on the window. `max_speed_km_s` bounds the range rate. |
| `check_sink(cls, params)` | Unknown settings are refused. A push sink writes the files it plans, with the same bytes however the stream is cut. A serving sink serves a short stream to a client and stops at its end. |

`load_plugin(entry)` loads a path, a name or a `plugins:` entry as a scenario would, and returns the module. A class
can be given as the class or as its registered name. The checks need no scenario; `seed`, `seconds` and tolerances
are keyword arguments (`power_tol_db`, `atol`, `event_s`).

`tests/test_conformance.py` runs these checks on the core modules and on the three examples.

## Packaging

A single file is enough for a scenario kept in one place. To share plugins, make a package and declare one **entry
point** per plugin in the group `assist.plugins`. The entry-point name is what a scenario lists; its value is the
module that registers the classes.

```toml
[project.entry-points."assist.plugins"]
pretty_gfsk = "assist_examples.pretty_gfsk"
tethered = "assist_examples.jitter_platform"
tcp = "assist_examples.tcp_sink"
```

```yaml
plugins:
  - pretty_gfsk
  - tcp
```

[`examples/assist-examples/`](examples/assist-examples) is such a package, with the same three modules as the loose
files. Install it into the environment that has ASSIST with `uv pip install -e examples/assist-examples`.

Entry points are used only to find the module of a name that a scenario lists. They are never scanned or imported
otherwise. Keep one plugin per module, or use `only:` in the scenario, so that a scenario loads no more than it
names. If two installed packages provide the same entry-point name, loading that name is an error that names both.

## The API version

`assist.api.API_VERSION` is `"major.minor"`, currently `"1.0"`. It covers:

- the names exported by `assist.api` (listed below);
- the rules in this guide (absolute indices, keyed randomness, unit power, the sink lifecycle, the capabilities);
- two kernels of `self.dsp`: `random_bytes(key, block, n)` and `white(key, h0, nhops, hop)`.

The other kernels (`shape`, `chirp`, `nco`, `synth` and so on) and every module other than `assist.api` and
`assist.testing` are internal. They may change without a version change.

Every plugin module declares the version it was written for, at its top level:

```python
ASSIST_API = "1.0"
```

| Plugin declares | ASSIST has | Result |
|---|---|---|
| same major, minor not above | | loads |
| same major, higher minor | an older minor | error: update ASSIST |
| another major | | error: the plugin has to be adapted |
| nothing | | error: add the line |

The minor version grows when something is added; plugins written for an older minor keep working. The major version
grows when a change can break a plugin.

## What is in `assist.api`

| Name | Use |
|---|---|
| `API_VERSION` | The version of this interface. |
| `register_signal`, `register_platform`, `register_sink` | The class decorators. |
| `Signal`, `SignalParams` | Base classes of a signal and of its parameter model. |
| `ShapedSignal`, `ShapedParams`, `rrc_taps` | Linear modulations with a root-raised-cosine pulse. |
| `Event` | One scheduled event: `start`, `end`, `params`. |
| `Keys`, `RngScope` | A signal's scope, and the scope type in general. |
| `unpack`, `int_range`, `float_range` | Bytes to bits; a parameter given as a number or an inclusive `[lo, hi]` range. |
| `Platform`, `PlatformParams`, `PlatformContext` | Base classes of a platform and its parameters; what it is built from. |
| `Geometry`, `Pass` | What `geometry()` and `passes()` return. |
| `Sink`, `SinkParams`, `SinkContext`, `OutFile` | A push sink, its settings, what it is built from, a file it writes. |
| `ServingSink`, `ServingParams` | A serving sink and the settings every serving sink has. |
| `StreamSink`, `StreamServer`, `RenderAhead` | The shared TCP serving code. |
