"""`run.json` (plan §7.2): provenance next to every render or serving session. Not truth: truth is computed from the
scenario.

It holds the resolved scenario (so `Scene` can be rebuilt from it), its hash, the package version and engine, the
render window as absolute sample indices, the noise source and ADC gain, and each output file's first sample.
"""

import json
from typing import TYPE_CHECKING, Any

from assist import __version__
from assist.timebase import format_iso

if TYPE_CHECKING:
    from pathlib import Path

    from assist.config.resolve import Resolved
    from assist.engine.render import Renderer
    from assist.sinks.wavfile import OutFile

ENGINE = "python reference (M3)"


def run_info(
    r: Resolved, rd: Renderer, fmt: str, n0: int, n: int | None, files: list[OutFile], status: str, **extra: Any
) -> dict[str, Any]:
    """n None: an endless stream (`serve` without --duration)."""
    fs = rd.fs
    return {
        "status": status,
        "assist_version": __version__,
        "engine": ENGINE,
        "scenario_hash": r.hash,
        "clock_start": r.scenario.clock.start,
        "render": {
            "format": fmt,
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
        "files": [f.info(fs) for f in files],
        **extra,
        "scenario": json.loads(r.canonical),
    }


def write_run_json(path: Path, info: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(info, indent=1, ensure_ascii=False) + "\n")
    tmp.replace(path)
