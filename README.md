# ASSIST

**Autonomous Satellite Signal Imitator Station.** ASSIST synthesises the 2.4 MS/s IQ stream of a VHF station
(RTL-SDR v3 at 137.000 MHz) and serves it over rtl_tcp or writes it to files, so that
ADRIFT can be tested against a sky whose content is known exactly.

- Implementation plan: [`samples/plan.md`](samples/plan.md)
- Progress, findings and decisions per milestone: [`checkpoint.md`](checkpoint.md)

Status: M0 (skeleton). The `assist` CLI has no commands yet.

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

Large local data (recordings, waterfalls, element snapshots) lives in `samples/`, which is gitignored except for
the plan. Tests never need it.
