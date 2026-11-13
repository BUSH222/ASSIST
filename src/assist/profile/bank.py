"""Sample bank. Clips of real signals from the recording, mixed to baseband and decimated to 50 kS/s.

Each clip is a complex64 .npy plus an entry in index.json: id, label guess, centre frequency, rate, start (Unix
and seconds from the file start), duration, level. The bank is local data (not committed).
"""

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from assist.profile.dsp import channelize

if TYPE_CHECKING:
    from pathlib import Path

    from assist.profile.wavio import Recording

RATE = 50_000
DECIM = 48


def clip(rec: Recording, t0_s: float, t1_s: float, f_centre_hz: float) -> np.ndarray:
    n0 = max(0, int(t0_s * rec.fs))
    n1 = min(rec.n_samples, int(t1_s * rec.fs))
    return channelize(rec.iq, n0, n1, rec.fs, f_centre_hz - 137.0e6, DECIM)


class Bank:
    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict[str, Any]] = []

    def add(
        self, rec: Recording, cid: str, label: str, t0_s: float, t1_s: float, f_centre_hz: float, **meta: object
    ) -> None:
        x = clip(rec, t0_s, t1_s, f_centre_hz)
        np.save(self.dir / f"{cid}.npy", x)
        p = np.abs(x) ** 2
        self.entries.append(
            {
                "id": cid,
                "label": label,
                "file": f"{cid}.npy",
                "centre_hz": f_centre_hz,
                "rate": RATE,
                "start_unix": rec.start_unix + t0_s,
                "start_s": t0_s,
                "duration_s": len(x) / RATE,
                "power_p50_p99_db": [
                    float(10 * np.log10(np.percentile(p, 50))),
                    float(10 * np.log10(np.percentile(p, 99))),
                ],
                "source": rec.path.name,
                **meta,
            }
        )

    def write_index(self) -> None:
        (self.dir / "index.json").write_text(json.dumps({"rate": RATE, "clips": self.entries}, indent=1))


def strongest_bursts(
    starts: np.ndarray, power: np.ndarray, frame_s: float, n: int, min_gap_s: float = 0.5
) -> list[tuple[float, float]]:
    """(t0, t1) seconds of the n strongest bursts in a fine-time channel power series (10 dB over the 20th pct)."""
    from assist.profile.dsp import hysteresis, runs

    noise = float(np.percentile(power, 20))
    exc = 10 * np.log10(np.maximum(power, 1e-30) / noise)
    ev = runs(hysteresis(exc, 10.0, 5.0))
    if len(ev) == 0:
        return []
    peak = np.array([exc[s:e].max() for s, e in ev])
    out: list[tuple[float, float]] = []
    for j in np.argsort(peak)[::-1]:
        s, e = ev[j]
        t0 = float(starts[s]) - 0.005
        if all(abs(t0 - o[0]) > min_gap_s for o in out):
            out.append((t0, float(starts[min(e, len(starts) - 1)]) + frame_s + 0.005))
        if len(out) >= n:
            break
    return sorted(out)
