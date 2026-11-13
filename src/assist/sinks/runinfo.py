"""`run.json` (plan §7.2): provenance next to every render or serving session. Not truth: truth is computed from the
scenario.

It holds the resolved scenario (so `Scene` can be rebuilt from it), its hash, the package version and engine, the
external modules it loaded, the stream window as absolute sample indices, the noise source and ADC gain, the sinks
that ran (their settings and what they report: file format, waterfall levels, stream statistics), and each output
file's first sample.
"""

import json
from typing import TYPE_CHECKING, Any

from assist import __version__, plugins
from assist.timebase import format_iso

if TYPE_CHECKING:
    from pathlib import Path

    from assist.config.resolve import Resolved
    from assist.engine.render import Renderer
    from assist.sinks.wavfile import OutFile


def engine_name(rd: Renderer) -> str:
    """The kernels behind a render: "c++ (M7)" (with "-march=native" when built so), or "numpy reference (M7)"."""
    if rd.dsp.__name__ == "assist.dsp_ref":
        return "numpy reference (M7)"
    return "c++, -march=native (M7)" if getattr(rd.dsp, "_native", False) else "c++ (M7)"


def run_info(
    r: Resolved,
    rd: Renderer,
    sinks: list[dict[str, Any]],
    n0: int,
    n: int | None,
    files: list[OutFile],
    status: str,
    **extra: Any,
) -> dict[str, Any]:
    """`sinks`: one {"kind", "params", "info"} per sink of the run. n None: an endless stream (a serving sink without
    --duration)."""
    fs = rd.fs
    return {
        "status": status,
        "assist_version": __version__,
        "engine": engine_name(rd),
        "scenario_hash": r.hash,
        "clock_start": r.scenario.clock.start,
        "api_version": plugins.API_VERSION,
        "plugins": [
            {
                "plugin": p.label,
                "version": p.version,
                "api": p.api,
                "sha256": p.sha256,
                "module": p.module.__name__,
                "file": getattr(p.module, "__file__", None),
                "distribution": p.distribution,
                "registers": [f"{cat} {name}" for cat, name in p.registered],
            }
            for p in plugins.active()
        ],
        "render": {
            "fc_hz": rd.fc_hz,
            "fs_hz": fs,
            "start_sample_abs": n0,
            "start_utc": format_iso(n0 / fs, 6),
            "n_samples": n,
            "duration_s": None if n is None else n / fs,
            "station_tz": r.scenario.station.tz,
        },
        "noise": {
            "source": rd.noise_spec.source,
            "sigma_counts": rd.noise_spec.sigma_counts,
            "adc_gain_counts_per_sqrt_mw": rd.gain,
            "t_sys_k": rd.scene.t_sys_k,
        },
        "emitters": [{"id": er.id, "narrow_rate_hz": er.rate, "f_bin_hz": er.f_bin_mhz / 1000} for er in rd.emitters],
        "sinks": sinks,
        "files": [f.info(fs) for f in files],
        **extra,
        "scenario": json.loads(r.canonical),
    }


def write_run_json(path: Path, info: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(info, indent=1, ensure_ascii=False) + "\n")
    tmp.replace(path)
