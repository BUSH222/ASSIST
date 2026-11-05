"""Truth (plan §8): T-DET across machines (golden file), T-KEY, consistency; catalog; CLI commands.

The golden truth was written on the dev Mac (`ASSIST_UPDATE_GOLDEN=1 uv run pytest tests/test_truth.py`); CI on
ubuntu must reproduce it byte for byte.
"""

import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from assist import truth as T
from assist.cli import app
from assist.config import loader
from assist.config.resolve import resolve, resolve_document
from assist.config.schema import CatalogOrbit
from assist.engine.scene import Scene
from assist.orbits import catalog as cat
from assist.orbits import elements as E
from assist.platforms.satellite import SatellitePlatform
from assist.timebase import parse_iso

ROOT = Path(__file__).resolve().parent.parent
BASIC = ROOT / "tests" / "fixtures" / "scenario_basic.yaml"
GOLDEN = ROOT / "tests" / "fixtures" / "truth_golden.csv"
GOLDEN_EVENTS = ROOT / "tests" / "fixtures" / "truth_golden.events.csv"
runner = CliRunner()


@pytest.fixture(autouse=True)
def _cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)


@pytest.fixture(scope="module")
def scene() -> Scene:
    return Scene(resolve(BASIC, base=ROOT).scenario)


def _csv(rows: list[dict[str, str]], cols: tuple[str, ...]) -> str:
    f = io.StringIO()
    T.write_csv(rows, cols, f)
    return f.getvalue()


def test_truth_matches_golden(scene: Scene) -> None:
    """T-DET (truth): the same text on every machine (frequencies to 1 mHz, angles 1e-6 deg, range 1 mm)."""
    table = T.compute(scene, scene.t_start, scene.t_end, 10.0)
    text, events = _csv(table.rows, T.COLUMNS), _csv(table.events, T.EVENT_COLUMNS)
    if os.environ.get("ASSIST_UPDATE_GOLDEN"):
        GOLDEN.write_text(text)
        GOLDEN_EVENTS.write_text(events)
    assert text == GOLDEN.read_text()
    assert events == GOLDEN_EVENTS.read_text()
    sources = {r["source_id"] for r in table.rows}
    assert sources == {"fm117", "m2-3", "z1"}  # kep only emits events (poisson)
    assert table.events and {e["source_id"] for e in table.events} == {"kep"}


def test_truth_is_window_independent(scene: Scene) -> None:
    a = T.compute(scene, scene.t_start, scene.t_end, 10.0)
    fresh = Scene(resolve(BASIC, base=ROOT).scenario)
    t1 = scene.t_start + 1500.0
    b = T.compute(fresh, t1, scene.t_end, 10.0)
    assert b.rows == [r for r in a.rows if parse_iso(r["t_utc"]) >= t1]


def test_frequencies_add_up_exactly(scene: Scene) -> None:
    for r in T.compute(scene, scene.t_start, scene.t_end, 30.0).rows:
        mhz = {k: round(float(r[k]) * 1000) for k in ("f_tx_hz", "f_rx_hz", "doppler_hz")}
        assert mhz["f_rx_hz"] == mhz["f_tx_hz"] + mhz["doppler_hz"]
        assert float(r["el_deg"]) >= 0


def test_zombie_truth_follows_the_carrier_model(scene: Scene) -> None:
    z = scene.source("z1").emitters[0]
    (p,) = z.platform.passes(scene.t_start, scene.t_end)
    assert p.aos_key is not None
    t = np.arange(float(p.aos_key), p.los, 1.0)
    tr = z.truth(t, 1.0)
    off, drift = z.carrier.pass_terms(int(p.aos_key))
    np.testing.assert_allclose(tr.f_tx_hz - z.carrier.rest_mhz / 1000, off / 1000 + drift * (t - p.aos_key), atol=1e-6)
    assert set(np.round(tr.active_fraction, 6)) >= {0.0, 1.0}
    on = tr.active_fraction == 1.0
    assert 0.1 < on.mean() < 0.3  # 40 / 220


def test_adding_a_source_changes_nothing_else(tmp_path: Path) -> None:
    """T-KEY: another source leaves every other source's resolved config and truth unchanged."""
    doc = loader.load_scenario(BASIC)
    extra = loader.load_scenario(BASIC)
    extra["sources"].insert(
        0,
        loader.expand_source(
            {
                "id": "a-new",
                "preset": "zombie",
                "orbit": {"synthetic_pass": {"aos": "+5min", "max_el_deg": 60}},
                "freq_hz": "uniform(136.1e6, 137.9e6)",
            }
        ),
    )
    r1, r2 = resolve_document(doc, ROOT), resolve_document(extra, ROOT)
    s1 = {s.id: s.model_dump() for s in r1.scenario.sources}
    s2 = {s.id: s.model_dump() for s in r2.scenario.sources}
    assert all(s1[k] == s2[k] for k in s1)
    sc1, sc2 = Scene(r1.scenario), Scene(r2.scenario)
    t1 = T.compute(sc1, sc1.t_start, sc1.t_end, 10.0)
    t2 = T.compute(sc2, sc2.t_start, sc2.t_end, 10.0)
    assert t1.rows == [r for r in t2.rows if r["source_id"] != "a-new"]
    assert t1.events == [e for e in t2.events if e["source_id"] != "a-new"]


def test_catalog_entries() -> None:
    sc = resolve(BASIC, base=ROOT).scenario
    full = {int(e["NORAD_CAT_ID"]): e for e in cat.entries(sc, ROOT)}
    only = {int(e["NORAD_CAT_ID"]) for e in cat.entries(sc, ROOT, sources_only=True)}
    kep = next(s for s in sc.sources if s.id == "kep")
    assert kep.catalog_norad is not None
    assert only == {41188, 57166, kep.catalog_norad}
    assert set(full) == only | {25984}  # + FM36 from the store
    m23 = next(s for s in sc.sources if s.id == "m2-3")
    assert isinstance(m23.orbit, CatalogOrbit) and m23.orbit.perturb is not None
    assert full[57166] == m23.orbit.omm  # published, not perturbed
    # Synthetic in-catalog object: its exported elements reproduce the truth orbit
    sat_cat = E.satellite_from_omm(full[kep.catalog_norad])
    plat = Scene(sc).source("kep").platform
    assert isinstance(plat, SatellitePlatform)
    sat_truth = plat.sat
    t = E.timescale().utc(2026, 11, 1, 18)
    d = np.asarray(sat_cat.at(t).position.km) - np.asarray(sat_truth.at(t).position.km)
    assert np.linalg.norm(d) < 1e-3
    # Leave-one-out removes the object even though the store has it
    sc2 = resolve(BASIC, sets=["sources[fm117].in_catalog=false"], base=ROOT).scenario
    assert 41188 not in {int(e["NORAD_CAT_ID"]) for e in cat.entries(sc2, ROOT)}


def test_cli_commands(tmp_path: Path) -> None:
    r = runner.invoke(app, ["validate", str(BASIC)])
    assert r.exit_code == 0 and "4 sources" in r.output
    out = tmp_path / "res.json"
    r = runner.invoke(app, ["resolve", str(BASIC), "-o", str(out), "--set", "sources[z1].freq_hz=137.35e6"])
    assert r.exit_code == 0, r.output
    res = json.loads(out.read_text())
    z1 = next(s for s in res["sources"] if s["id"] == "z1")
    assert z1["emitters"][0]["freq_hz"] == 137.35e6
    r = runner.invoke(app, ["passes", str(out), "--to", "+100min"])
    assert r.exit_code == 0 and "ORBCOMM FM117" in r.output and "z1" in r.output
    tcsv = tmp_path / "t.csv"
    r = runner.invoke(app, ["truth", str(out), "-o", str(tcsv), "--from", "+2min", "--to", "+3min"])
    assert r.exit_code == 0, r.output
    lines = tcsv.read_text().splitlines()
    assert lines[0] == ",".join(T.COLUMNS) and len(lines) == 61  # z1 visible for the whole minute
    tj = tmp_path / "t.jsonl"
    r = runner.invoke(app, ["truth", str(out), "-o", str(tj), "--format", "jsonl", "--to", "+100min", "--step", "60s"])
    assert r.exit_code == 0, r.output
    rows = [json.loads(x) for x in tj.read_text().splitlines()]
    assert rows and isinstance(rows[0]["f_rx_hz"], float) and rows[0]["in_catalog"] in (True, False)
    assert (tmp_path / "t.events.jsonl").is_file()
    for suffix in ("tle", "json"):
        r = runner.invoke(app, ["catalog", str(out), "-o", str(tmp_path / f"c.{suffix}")])
        assert r.exit_code == 0, r.output
    assert len(json.loads((tmp_path / "c.json").read_text())) == 4
    assert len((tmp_path / "c.tle").read_text().splitlines()) == 12
    r = runner.invoke(app, ["schema"])
    assert r.exit_code == 0 and json.loads(r.output)["title"] == "Scenario"
    r = runner.invoke(app, ["validate", str(BASIC), "--set", "sources[z1].level.cal=nope"])
    assert r.exit_code == 2 and "unknown calibration" in r.output
    r = runner.invoke(app, ["elements", "--help"])
    assert r.exit_code == 0 and "fetch" in r.output
