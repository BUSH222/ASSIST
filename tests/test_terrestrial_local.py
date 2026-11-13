"""Local checks of the terrestrial environment (need samples/: the element snapshots of
scenarios/terrestrial_hour.yaml).

Closed loop on VDL: render two minutes of the recording's hour (scenarios/terrestrial_hour.yaml, 16:50 UTC) and run
the same `assist profile terrestrial` measurement on ASSIST's own uint8 stream that produced
profiles/vhf137/terrestrial.json from the recording. The fixed stations come back at their measured levels, the strong
ones clip as much as in the recording (above the recording's background rail fraction), and the aircraft bursts have
the recording's rate and level spread.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from assist.config.resolve import resolve
from assist.engine.render import Renderer
from assist.engine.scene import Scene
from assist.profile import terrestrial as ter
from assist.timebase import parse_iso, sample_at
from assist.waterfall import iq_from_u8

ROOT = Path(__file__).resolve().parent.parent
FS = 2_400_000
DUR = 120
PROFILE = ROOT / "profiles" / "vhf137"

pytestmark = pytest.mark.skipif(
    not (ROOT / "samples" / "active.csv").is_file(), reason="needs samples/ (element snapshots)"
)


class _Stream:
    """A rendered window with the reader interface of profile.wavio.Recording."""

    fs = FS

    def __init__(self, u8: np.ndarray, t0: float) -> None:
        self._u, self.start_unix = u8, t0

    def u8(self, a: int, n: int) -> np.ndarray:
        a = max(0, a)
        return np.array(self._u[2 * a : 2 * (a + n)])

    def iq(self, a: int, n: int) -> np.ndarray:
        x = np.zeros(n, np.complex64)
        lo, hi = max(0, a), min(len(self._u) // 2, a + n)
        if hi > lo:
            x[lo - a : hi - a] = iq_from_u8(self._u[2 * lo : 2 * hi])
        return x


@pytest.fixture(scope="module")
def measured(monkeypatch_module: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    monkeypatch_module.chdir(ROOT)
    t0 = parse_iso("2026-11-01T16:50:00Z")
    rd = Renderer(Scene(resolve(ROOT / "scenarios" / "terrestrial_hour.yaml", base=ROOT).scenario), ROOT)
    st = _Stream(rd.u8(sample_at(t0, FS), DUR * FS), t0)
    sigma = json.loads((PROFILE / "profile.json").read_text())["adc"]["noise_sigma_mad_counts"]
    chans = ter.channelize_fft(st, 0, DUR, [136.925e6, 136.975e6])
    n0 = {c: ter.assist_n0_counts(PROFILE, c) for c in chans}
    return {f"{c / 1e6:.3f}": ter.vdl_channel(st, 0.0, y, n0[c], sigma, lambda m: None) for c, y in chans.items()}


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_vdl_closed_loop(measured: dict[str, dict[str, Any]]) -> None:
    ref = json.loads((PROFILE / "terrestrial.json").read_text())["vdl"]
    for ch, got in measured.items():
        exp = ref[ch]
        assert abs(got["floor_over_assist_db"]) < 0.1  # ASSIST's floor is its own Gaussian noise
        assert len(got["fixed"]) == len(exp["fixed"]), ch
        bg = exp["background_rail_fraction"]
        for g, e in zip(got["fixed"], exp["fixed"], strict=True):
            assert abs(g["cn0_dbhz_measured"] - e["cn0_dbhz_measured"]) < 0.5, (ch, g, e)
            # clipping: the recording's rail fraction above its background
            want = max(0.0, (e["rail_fraction_p50"] - bg) / (1 - bg))
            assert abs(g["rail_fraction_p50"] - want) < 0.03 + 0.1 * want, (ch, g["rail_fraction_p50"], want)
            assert abs(g["rate_per_s"] / e["rate_per_s"] - 1) < 0.5, (ch, g["rate_per_s"], e["rate_per_s"])
    air, exp = measured["136.975"]["spread"], ref["136.975"]["spread"]
    assert abs(air["rate_per_s"] / exp["rate_per_s"] - 1) < 0.35
    p, q = air["cn0_dbhz_p10_p16_p50_p84_p90"], exp["cn0_dbhz_p10_p16_p50_p84_p90"]
    assert abs(p[2] - q[2]) < 1.5 and abs((p[3] - p[1]) - (q[3] - q[1])) < 2.5
