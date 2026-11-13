"""Full-pass check of a real render (local only, skipped without it, as in CI).

The render is `uv run assist run scenarios/zombie_single.yaml --sink wav` (into out/zombie_single; 18 min, two
WAV files). Its run.json holds the resolved scenario, so the truth is rebuilt from it, and the WAVs are read with
the profile tool's reader. Across the whole pass, in 1 s windows inside the zombie's on intervals where truth C/N0 >=
30 dB-Hz (AM and noise included):
- Doppler: tone frequency vs truth f_rx, <= 0.1 Hz RMS (mostly the estimator's own noise: 0.5 s windows give ~80 mHz,
  2 s windows ~11 mHz; the noiseless CI check shows ~1 mHz);
- level: C/N0 vs truth, mean within 0.1 dB, RMS within the windows' own scatter (<= 0.4 dB);
- off intervals: no carrier.
"""

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from assist import measure
from assist.config import schema as S
from assist.engine.scene import Scene
from assist.profile.wavio import Recording

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "zombie_single"
RUN = OUT / "run.json"


def _complete() -> bool:
    if not RUN.is_file():
        return False
    run = json.loads(RUN.read_text())
    return bool(run.get("status") == "complete" and all((OUT / f["name"]).is_file() for f in run["files"]))


pytestmark = pytest.mark.skipif(
    not _complete(), reason="needs out/zombie_single (assist run scenarios/zombie_single.yaml --sink wav)"
)

WIN_S = 1.0
MIN_CN0_DBHZ = 30.0


class _Render:
    def __init__(self) -> None:
        self.run = json.loads(RUN.read_text())
        self.fs = int(self.run["render"]["fs_hz"])
        self.fc = float(self.run["render"]["fc_hz"])
        self.files = [(f["start_sample_abs"], f["n_samples"], Recording(OUT / f["name"])) for f in self.run["files"]]
        self.scene = Scene(S.Scenario.model_validate(self.run["scenario"]))

    def iq(self, n0: int, n: int) -> np.ndarray | None:
        """Absolute samples [n0, n0 + n) as complex counts, or None if they span two files or lie outside."""
        for start, count, rec in self.files:
            if start <= n0 and n0 + n <= start + count:
                return measure.iq_from_u8(rec.u8(n0 - start, n))
        return None


@pytest.fixture(scope="module")
def rendered() -> _Render:
    return _Render()


def test_full_pass_doppler_and_level(rendered: _Render) -> None:
    sc, fs = rendered.scene, rendered.fs
    src = sc.sources[0]
    em = src.emitters[0]
    (p,) = src.platform.passes(sc.t_start, sc.t_end)
    on = em.schedule.on_intervals(p.aos, p.los)  # ty: ignore[unresolved-attribute]
    n = round(WIN_S * fs)
    f_err, lvl_err, cn0s = [], [], []
    for a, b in on:
        for t0 in np.arange(max(a, p.aos) + 0.1, min(b, p.los) - WIN_S - 0.1, WIN_S):
            n0 = round(t0 * fs)
            y = rendered.iq(n0, n)
            if y is None:
                continue
            tm = (n0 + np.arange(0, n, fs // 1000) + fs // 2000) / fs
            tr = em.truth(tm, 1e-3)
            cn0 = 10 * np.log10(np.mean(10 ** (tr.cn0_dbhz / 10)))
            if cn0 < MIN_CN0_DBHZ:
                continue
            f_rx = tr.f_rx_hz - rendered.fc
            f_mid = float(em.truth(np.array([(n0 + n / 2) / fs]), 1e-3).f_rx_hz[0]) - rendered.fc
            f_err.append(measure.tone_freq(y, fs, f_mid, half_width=40.0) - f_mid)
            got, _c, _n0 = measure.cn0_dbhz(y, fs, float(f_rx.min()) - 20, float(f_rx.max()) + 20, ring=(300, 3000))
            lvl_err.append(got - cn0)
            cn0s.append(cn0)
    f_err, lvl_err = np.array(f_err), np.array(lvl_err)
    print(
        f"\n{len(f_err)} windows, truth C/N0 {min(cn0s):.1f}..{max(cn0s):.1f} dB-Hz; "
        f"frequency error RMS {np.sqrt(np.mean(f_err**2)) * 1000:.1f} mHz (max {np.abs(f_err).max() * 1000:.1f}); "
        f"C/N0 error mean {lvl_err.mean():+.3f} dB, RMS {np.sqrt(np.mean(lvl_err**2)):.3f} dB"
    )
    assert len(f_err) >= 60
    assert np.sqrt(np.mean(f_err**2)) <= 0.1  # Doppler over the pass
    assert abs(lvl_err.mean()) <= 0.1  # level over the pass
    assert np.sqrt(np.mean(lvl_err**2)) <= 0.4


def test_off_intervals_are_silent(rendered: _Render) -> None:
    sc, fs = rendered.scene, rendered.fs
    src = sc.sources[0]
    em = src.emitters[0]
    (p,) = src.platform.passes(sc.t_start, sc.t_end)
    on = em.schedule.on_intervals(p.aos, p.los)  # ty: ignore[unresolved-attribute]
    checked = 0
    for (_a0, b0), (a1, _b1) in pairwise(on):
        t0 = (b0 + a1) / 2
        n0, n = round(t0 * fs), fs
        y = rendered.iq(n0, n)
        if y is None:
            continue
        f = float(em.truth(np.array([t0 + 0.5]), 1e-3).f_rx_hz[0]) - rendered.fc
        with np.errstate(invalid="ignore"):
            _cn0, c, n0d = measure.cn0_dbhz(y, fs, f - 40, f + 40, ring=(300, 3000))
        assert c < 10**1.5 * n0d  # below 15 dB-Hz (on: 33-41): no carrier
        checked += 1
    assert checked >= 2
