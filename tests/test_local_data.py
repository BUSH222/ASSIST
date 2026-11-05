"""Checks on the local data in samples/ (skipped without it, as in CI).

M2 done-when: modelled METEOR levels reproduce the waterfall passes within M1's residual.

The passes are re-measured with the M1 code (P10: band SNR per row). The model goes through the simulator's path:
a scenario per waterfall day with both METEORs at their calibrated levels (`cal: meteor_m2_3` / `meteor_m2_4`),
resolved and built into a scene (elements nearest the day, spline geometry, link budget, T_sys). Rows are compared
as in M1's fit: 10 log10(1 + SNR), elevation >= 3 deg, passes with max elevation > 5 deg. Skipped without samples/.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from assist.config.resolve import resolve
from assist.engine.scene import Scene
from assist.orbits.elements import load_newest
from assist.profile import meteor_cal
from assist.profile.rawio import open_raw
from assist.station import load_station
from assist.timebase import format_iso

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
RAWS = sorted(SAMPLES.glob("waterfall_*.raw"))
METEORS = {57166: ("m2-3", "meteor_m2_3"), 59051: ("m2-4", "meteor_m2_4")}

pytestmark = pytest.mark.skipif(not RAWS or not (SAMPLES / "station.yaml").is_file(), reason="needs local samples/")


def _scenario(tmp: Path, t0: float, t1: float) -> Path:
    doc = {
        "seed": 0,
        "clock": {"start": format_iso(math.floor(t0)), "duration": math.ceil(t1 - math.floor(t0)) + 1},
        "profile": "profiles/vhf137",
        "elements": {"files": ["samples/active.csv", "samples/satnogs.csv"]},
        "sources": [
            {
                "id": sid,
                "orbit": {"norad": norad},
                "emitters": [{"id": "lrpt", "signal": {"kind": "oqpsk"}, "freq_hz": 137.9e6, "level": {"cal": cal}}],
            }
            for norad, (sid, cal) in METEORS.items()
        ],
    }
    p = tmp / f"meteor_{int(t0)}.json"
    p.write_text(json.dumps(doc))
    return p


def test_modelled_meteor_levels_match_waterfall_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    mc = json.loads((ROOT / "profiles/vhf137/meteor_cal.json").read_text())
    frac = float(mc["band_power_fraction"])
    st = load_station(SAMPLES / "station.yaml")
    els = load_newest([SAMPLES / "active.csv", SAMPLES / "satnogs.csv"])
    res: dict[int, list[np.ndarray]] = {n: [] for n in METEORS}
    for path in RAWS:
        raw = open_raw(path)
        t0, t1 = float(raw.row_time(0)), float(raw.row_time(raw.n_rows - 1))
        r = resolve(_scenario(tmp_path, t0, t1))
        assert r.scenario.station.source == "samples/station.yaml"
        scene = Scene(r.scenario)
        for pd in meteor_cal.collect([raw], els, st, tuple(METEORS)):
            if pd.p.max_el_deg <= 5:
                continue
            em = scene.source(METEORS[pd.norad][0]).emitters[0]
            g = em.platform.geometry(pd.t)
            _rx, cn0 = em.level(pd.t, g.el_deg, g.az_deg, g.range_m, g.off_nadir_deg)
            snr_model_db = cn0 - 10 * np.log10(meteor_cal.HALF_BW * 2) + 10 * np.log10(frac)
            ok = pd.el >= 3.0
            meas = 10 * np.log10(np.maximum(1 + pd.snr_lin[ok], 1e-3))
            model = 10 * np.log10(1 + 10 ** (snr_model_db[ok] / 10))
            res[pd.norad].append(meas - model)
    rms = {n: float(np.sqrt(np.mean(np.concatenate(v) ** 2))) for n, v in res.items()}
    allr = np.concatenate([np.concatenate(v) for v in res.values()])
    total = float(np.sqrt(np.mean(allr**2)))
    summary = {
        "rows": int(allr.size),
        "rms_db": total,
        "median_db": float(np.median(allr)),
        "mean_db": float(np.mean(allr)),
        "rms_db_per_satellite": {METEORS[n][1]: v for n, v in rms.items()},
        "m1_rms_db": mc["fit_both"]["residual_rms_db"],
    }
    print(json.dumps(summary))  # numbers for the checkpoint (pytest -s)
    assert allr.size == mc["fit_both"]["rows"]
    assert total <= mc["fit_both"]["residual_rms_db"]
    for n, (_sid, _cal) in METEORS.items():
        name = "METEOR-M2 3" if n == 57166 else "METEOR-M2 4"
        assert rms[n] <= mc["fit_per_satellite"][name]["residual_rms_db"] + 0.02
    assert abs(float(np.median(allr))) <= 0.5


def test_example_scenario_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    r = resolve(ROOT / "scenarios" / "example.yaml", now=1793551140.0)
    assert {s.id for s in r.scenario.sources} == {"m2-3", "m2-4", "transit-5b5", "z1"}
    assert r.scenario.station.source == "samples/station.yaml"
    assert Scene(r.scenario).source("transit-5b5").platform.norad == 965
