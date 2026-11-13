# assist-examples

The three example plugins of ASSIST as an installable package. The modules in `src/assist_examples/` are the same
files as the loose ones in [`examples/plugins/`](../plugins); only the way a scenario names them differs.

| Entry point | Module | Registers |
|---|---|---|
| `pretty_gfsk` | `assist_examples.pretty_gfsk` | signal kind `pretty_gfsk` (38.4 kbaud GFSK bursts) |
| `tethered` | `assist_examples.jitter_platform` | platform `tethered` (fixed direction, swinging range) |
| `tcp` | `assist_examples.tcp_sink` | sink `tcp` (raw TCP server: cu8, cs16 or cf32) |

## Install

Into the environment that has ASSIST, from the repository root:

```sh
uv pip install -e examples/assist-examples
```

## Use

A scenario lists the entry-point names it needs. Only those modules are imported:

```yaml
plugins:
  - pretty_gfsk
  - tcp
sinks:
  - tcp: {port: 5555, format: cs16}
sources:
  - id: pretty
    orbit: {norad: 58023}
    emitters:
      - id: tm
        signal: {kind: pretty_gfsk}
        freq_hz: 137.78e6
        schedule: {type: exp_gaps, mean_gap_s: 6, min_gap_s: 1}
        level: {cn0_zenith_dbhz: 62}
```

`assist plugins list scenario.yaml` shows what the scenario loads and where each name comes from. `run.json` records
the distribution name and version of every plugin that was loaded.

The same modules work without installing anything, as files:

```yaml
plugins:
  - path: examples/plugins/pretty_gfsk.py
```

## As a template

To start a package of your own: copy this directory, rename the package and the distribution, replace the modules,
and list one entry point per module in `pyproject.toml`. Every module needs `ASSIST_API = "1.0"` at its top level.
The guide is [`PLUGINS.md`](../../PLUGINS.md) in the repository root.
