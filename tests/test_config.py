"""Scenario loading, random expressions, schema and resolve (plan §3.3); T-DET (resolve part)."""

import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from assist.config import loader, randexpr
from assist.config.loader import ConfigError
from assist.config.resolve import Resolved, resolve, resolve_document
from assist.config.schema import CatalogOrbit, OnOffSchedule, Scenario, SyntheticPassOrbit
from assist.emitters.link import t_sys_k
from assist.rng import KeyedRng
from assist.station import FALLBACK_ALT_M, FALLBACK_LAT_DEG, FALLBACK_LON_DEG

ROOT = Path(__file__).resolve().parent.parent
BASIC = ROOT / "tests" / "fixtures" / "scenario_basic.yaml"
NOW = 1793551140.7  # 2026-11-01T16:39:00.7Z


@pytest.fixture(autouse=True)
def _cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)


def _write(tmp: Path, name: str, text: str) -> Path:
    p = tmp / name
    p.write_text(text)
    return p


def _doc(**over: object) -> dict[str, Any]:
    d = {
        "seed": 1,
        "clock": {"start": "2026-11-01T16:39:00Z", "duration": 600},
        "station": {"file": None, "lat": 55.5, "lon": 37.1, "alt_m": 172},
        "sources": [
            {
                "id": "z1",
                "preset": "zombie",
                "orbit": {"synthetic_pass": {"aos": "+2min", "max_el_deg": 30, "direction": "asc", "side": "west"}},
                "freq_hz": 137.5e6,
            }
        ],
    }
    d.update(over)
    return d


def _resolve_doc(d: dict[str, Any], tmp: Path) -> Resolved:
    return resolve(_write(tmp, "s.json", json.dumps(d)))


# --- YAML and merging ---


def test_yaml_reads_exponent_floats_and_keeps_timestamps() -> None:
    d = loader.parse_yaml("a: 137.0e6\nb: 1e6\nc: 2026-11-01T16:39:00Z\nd: '+03:00'\ne: 7\n")
    assert d == {"a": 137e6, "b": 1e6, "c": "2026-11-01T16:39:00Z", "d": "+03:00", "e": 7}
    assert isinstance(d["b"], float)


def test_deep_merge_rules() -> None:
    base = {
        "a": {"x": 1, "y": 2},
        "l": [1, 2],
        "s": [{"id": "a", "v": 1}, {"id": "b", "v": 2}],
        "sch": {"type": "on_off", "on_s": 40},
    }
    over = {"a": {"y": 3}, "l": [9], "s": [{"id": "b", "v": 5}, {"id": "c", "v": 6}], "sch": {"type": "continuous"}}
    m = loader.deep_merge(base, over)
    assert m["a"] == {"x": 1, "y": 3}
    assert m["l"] == [9]
    assert m["s"] == [{"id": "a", "v": 1}, {"id": "b", "v": 5}, {"id": "c", "v": 6}]
    assert m["sch"] == {"type": "continuous"}  # discriminator changed: replaced, not merged
    assert base["a"] == {"x": 1, "y": 2}  # inputs untouched


def test_includes_merge_relative_then_package(tmp_path: Path) -> None:
    _write(tmp_path, "inc.yaml", "seed: 5\nclock: {duration: 60}\nsources: [{id: a, x: 1}]\n")
    main = _write(
        tmp_path, "main.yaml", "include: [inc.yaml, presets/zombie.yaml]\nclock: {start: now}\nsources: [{id: b}]\n"
    )
    d = loader.load_with_includes(main)
    assert d["seed"] == 5 and d["clock"] == {"duration": 60, "start": "now"}
    assert [s["id"] for s in d["sources"]] == ["a", "b"]
    assert d["platform"] == "satellite"  # found in the package (presets/zombie.yaml)
    _write(tmp_path, "c1.yaml", "include: [c2.yaml]\n")
    _write(tmp_path, "c2.yaml", "include: [c1.yaml]\n")
    with pytest.raises(ConfigError, match="cycle"):
        loader.load_with_includes(tmp_path / "c1.yaml")


def test_set_overrides_by_id_and_path() -> None:
    d = _doc()
    loader.apply_set(d, "sources[z1].freq_hz=137.35e6")
    loader.apply_set(d, "sources[0].orbit.synthetic_pass.max_el_deg=60")
    loader.apply_set(d, "frontend.lna.gain_db=22")
    loader.apply_set(d, "sources[z1].schedule.on_s=30")
    assert d["sources"][0]["freq_hz"] == 137.35e6
    assert d["sources"][0]["orbit"]["synthetic_pass"]["max_el_deg"] == 60
    assert d["frontend"] == {"lna": {"gain_db": 22}}
    with pytest.raises(ConfigError, match="no list item"):
        loader.apply_set(d, "sources[nope].freq_hz=1")
    with pytest.raises(ConfigError):
        loader.apply_set(d, "seed")


def test_preset_expansion_routes_keys() -> None:
    src = {
        "id": "z",
        "preset": "zombie",
        "orbit": {"norad": 965},
        "freq_hz": 136.5e6,
        "schedule": {"on_s": 30},
        "carrier": {"offset_hz": 12},
        "name": "TRANSIT",
    }
    out = loader.expand_source(src)
    (em,) = out["emitters"]
    assert out["platform"] == "satellite" and out["preset"] == "zombie" and out["name"] == "TRANSIT"
    assert em["freq_hz"] == 136.5e6 and em["offset_hz"] == 12
    assert em["schedule"] == {"type": "on_off", "on_s": 30, "off_s": 180, "ramp_ms": 5}
    with pytest.raises(ConfigError, match="unknown key"):
        loader.expand_source({**src, "frequency": 1})
    with pytest.raises(ConfigError, match="unknown preset"):
        loader.expand_source({"id": "x", "preset": "nope"})


def test_band_defaults_fill_frontend(tmp_path: Path) -> None:
    sc = _resolve_doc(_doc(frontend={"t_ant_k": 300}), tmp_path).scenario
    assert sc.frontend.t_ant_k == 300
    assert sc.frontend.fc_hz == 137e6 and sc.frontend.tuner_nf_db == 3.5
    assert sc.antenna.pattern.type == "poly"


# --- random expressions ---


@pytest.mark.parametrize(
    ("expr", "check"),
    [
        ("uniform(136.1e6, 137.9e6)", lambda v: isinstance(v, float) and 136.1e6 <= v <= 137.9e6),
        ("normal(0, 50)", lambda v: isinstance(v, float)),
        ("randint(7, 12)", lambda v: isinstance(v, int) and 7 <= v <= 12),
        ("choice([east, west])", lambda v: v in ("east", "west")),
        ("sample([1, 2, 3, 4], 2)", lambda v: len(v) == 2 and len(set(v)) == 2),
        ("bernoulli(0.5)", lambda v: isinstance(v, bool)),
    ],
)
def test_random_expressions(expr: str, check: object) -> None:
    v = randexpr.draw(expr, KeyedRng(3).stream("x"))
    assert check(v)  # ty: ignore[call-non-callable]


@pytest.mark.parametrize(
    "bad", ["uniform(2, 1)", "randint(1.5, 3)", "choice([])", "sample([1], 2)", "normal(0)", "bernoulli(2)"]
)
def test_bad_random_expressions(bad: str) -> None:
    with pytest.raises(ConfigError):
        randexpr.draw(bad, KeyedRng(3).stream("x"))


def test_randint_is_inclusive() -> None:
    g = KeyedRng(0).stream("x")
    assert {randexpr.draw("randint(7, 8)", g) for _ in range(200)} == {7, 8}


def test_draws_keyed_by_id_not_position() -> None:
    srcs = [{"id": i, "freq_hz": "uniform(0, 1)"} for i in ("a", "b", "c")]
    d1 = randexpr.draw_all({"sources": srcs}, KeyedRng(9))
    d2 = randexpr.draw_all({"sources": list(reversed(srcs[1:]))}, KeyedRng(9))
    v1 = {s["id"]: s["freq_hz"] for s in d1["sources"]}
    v2 = {s["id"]: s["freq_hz"] for s in d2["sources"]}
    assert v1["b"] == v2["b"] and v1["c"] == v2["c"]
    assert len(set(v1.values())) == 3


# --- schema ---


def test_schema_rejects_typos_duplicates_and_bad_levels(tmp_path: Path) -> None:
    for bad, match in [
        (_doc(clokc={}), "clokc"),
        (_doc(sources=[_doc()["sources"][0], _doc()["sources"][0]]), "duplicate source"),
        (_doc(sources=[{**_doc()["sources"][0], "level": {"p_tx_dbw": 1, "cal": "meteor"}}]), "exactly one"),
        (_doc(sources=[{**_doc()["sources"][0], "signal": {"kind": "fm"}}]), "unknown signal kind"),
        (_doc(sources=[{k: v for k, v in _doc()["sources"][0].items() if k != "freq_hz"}]), "freq_hz"),
    ]:
        with pytest.raises((ValidationError, ConfigError), match=match):
            _resolve_doc(bad, tmp_path)


def test_json_schema_exports() -> None:
    s = Scenario.model_json_schema()
    assert "sources" in s["properties"] and "Emitter" in s["$defs"]


# --- resolve ---


def test_resolve_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    """T-DET (resolve part): two resolves give identical JSON; resolving the result changes nothing."""
    a, b = resolve(BASIC, now=NOW), resolve(BASIC, now=NOW)
    assert a.canonical == b.canonical and a.hash == b.hash
    again = resolve(_write(tmp_path, "r.json", a.pretty()))
    assert again.canonical == a.canonical


def test_resolve_fills_everything(tmp_path: Path) -> None:
    sc = resolve(BASIC).scenario
    by = {s.id: s for s in sc.sources}
    fm = by["fm117"].orbit
    assert isinstance(fm, CatalogOrbit) and fm.omm is not None
    assert fm.omm["EPOCH"] == "2026-11-02T02:37:16.867200"  # nearest the clock start, not newest
    assert by["fm117"].in_catalog is True and by["z1"].in_catalog is False
    assert by["kep"].catalog_norad is not None and 90000 <= by["kep"].catalog_norad < 100000
    z = by["z1"]
    assert isinstance(z.orbit, SyntheticPassOrbit)
    sp = z.orbit.synthetic_pass
    assert sp.aos == "2026-11-01T16:41:00Z" and sp.solution is not None
    (em,) = z.emitters
    assert isinstance(em.schedule, OnOffSchedule)
    assert em.schedule.phase_s is not None and em.am is not None and em.am.phase_rad is not None
    assert em.lvl.p_tx_dbw is not None and "cn0_zenith_dbhz 45" in str(em.lvl.resolved_from)
    f = em.freqs_hz[0]
    assert round(f * 1000) == f * 1000  # quantised to 1 mHz
    assert sc.assist_version and sc.band_hash and sc.profile_hash is None
    # --set and seeds change draws; the scenario hash follows
    other = resolve(BASIC, sets=["seed=43"])
    assert other.hash != resolve(BASIC).hash


def test_clock_now_resolves_to_whole_second(tmp_path: Path) -> None:
    sc = resolve(_write(tmp_path, "s.json", json.dumps(_doc(clock={"start": "now"}))), now=NOW).scenario
    assert sc.clock.start == "2026-11-01T16:39:00Z"


def test_station_file_inline_and_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    st_file = _write(tmp_path, "st.yaml", "station_lat: 50.25\nstation_lon: 10.5\naltitude_m: 300\n")
    d = _doc()
    d["station"] = {"file": str(st_file)}
    s = _resolve_doc(d, tmp_path).scenario.station
    assert (s.lat, s.lon, s.alt_m, s.source, s.file) == (50.25, 10.5, 300.0, str(st_file), None)
    d["station"] = {"file": str(tmp_path / "missing.yaml")}
    s = _resolve_doc(d, tmp_path).scenario.station
    assert (s.lat, s.lon, s.alt_m, s.source) == (FALLBACK_LAT_DEG, FALLBACK_LON_DEG, FALLBACK_ALT_M, "fallback")
    d["station"] = {"lat": 1.0, "lon": 2.0}  # explicit position: the default file is not read
    s = _resolve_doc(d, tmp_path).scenario.station
    assert (s.lat, s.lon, s.source) == (1.0, 2.0, "config")
    # No station block: samples/station.yaml if present, else the fallback (CI)
    monkeypatch.chdir(tmp_path)
    d.pop("station")
    s = _resolve_doc(d, tmp_path).scenario.station
    assert (s.lat, s.lon, s.source) == (FALLBACK_LAT_DEG, FALLBACK_LON_DEG, "fallback")
    (tmp_path / "samples").mkdir()
    _write(tmp_path / "samples", "station.yaml", "station_lat: 50.25\nstation_lon: 10.5\n")
    s = _resolve_doc(d, tmp_path).scenario.station
    assert (s.lat, s.lon, s.alt_m, s.source) == (50.25, 10.5, FALLBACK_ALT_M, "samples/station.yaml")


def test_levels_cal_is_t_sys_invariant(tmp_path: Path) -> None:
    """A calibrated level keeps C/N0 whatever T_ant is: P_tx moves with 10 log10(T_sys)."""
    d = _doc()
    d["sources"][0]["level"] = {"cal": "meteor_m2_3"}
    p500 = _resolve_doc(d, tmp_path).scenario.sources[0].emitters[0].lvl.p_tx_dbw
    d2 = copy.deepcopy(d)
    d2["frontend"] = {"t_ant_k": 1000}
    p1000 = _resolve_doc(d2, tmp_path).scenario.sources[0].emitters[0].lvl.p_tx_dbw
    assert p500 is not None and p1000 is not None
    assert p1000 - p500 == pytest.approx(10 * math.log10(t_sys_k(1000, 1.465) / t_sys_k(500, 1.465)))
    assert p500 == pytest.approx(-2.227, abs=0.001)  # M1: METEOR-M2 3 at T_ant 500 K
    d["sources"][0]["level"] = {"cal": "nope"}
    with pytest.raises(ConfigError, match="unknown calibration"):
        _resolve_doc(d, tmp_path)


def test_resolve_document_accepts_merged_dicts() -> None:
    doc = loader.load_scenario(BASIC)
    assert resolve_document(doc, ROOT, NOW).hash == resolve(BASIC, now=NOW).hash
