"""Platforms by capability (PLAN.md R2): the registry, capability rules, and the external `tethered` platform."""

import itertools
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from assist import plugins, registry, truth
from assist.cli import app
from assist.config.loader import ConfigError, load_scenario
from assist.config.resolve import resolve, resolve_document
from assist.engine import workers
from assist.engine.render import Renderer
from assist.engine.scene import Scene
from assist.platforms.base import Platform, check_platform_params, platform_class, register_platform
from assist.timebase import parse_iso, sample_at

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = "examples/plugins/jitter_platform.py"
BASIC = ROOT / "tests/fixtures/scenario_basic.yaml"
FS = 2_400_000
C = 299_792_458.0
T0 = "2026-11-04T12:00:00Z"
F_HZ = 137.3e6
runner = CliRunner()


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.chdir(ROOT)
    yield
    plugins.activate([])


def _balloon(sid: str = "balloon", **kw: Any) -> dict[str, Any]:
    return {
        "id": sid,
        "platform": "tethered",
        "platform_params": {"az_deg": 200, "el_deg": 35, "range_m": 3000, "swing_m": 50, "period_ms": 5000},
        "emitters": [{"id": "beacon", "signal": {"kind": "cw"}, "freq_hz": F_HZ, "level": {"cn0_dbhz": 60}}],
        **kw,
    }


def _doc(sources: list[dict[str, Any]], declare: bool = True) -> dict[str, Any]:
    return {
        "seed": 9,
        "clock": {"start": T0, "duration": "10min"},
        "station": {"file": None, "lat": 55.5, "lon": 37.1, "alt_m": 172},
        **({"plugins": [{"path": PLUGIN}]} if declare else {}),
        "sources": sources,
    }


def _resolved(tmp_path: Path, doc: dict[str, Any]) -> Any:
    p = tmp_path / "scenario.yaml"
    p.write_text(yaml.safe_dump(doc))
    return resolve_document(load_scenario(p), ROOT)


# --- registry and capabilities ---


def test_core_platforms_and_their_capabilities() -> None:
    assert registry.platforms.names() == ["satellite", "ground", "local"]
    sat, ground, local = (platform_class(n) for n in ("satellite", "ground", "local"))
    assert (sat.kind, ground.kind, local.kind) == ("satellite", "ground", "local")
    assert sat.needs_orbit and sat.catalog and sat.supports_oscillator and sat.fades_at_horizon
    assert sat.level_mode == "link_budget" and not sat.fixed and not sat.allows_wideband
    for cls in (ground, local):
        assert cls.fixed and cls.level_mode == "received"
        assert not (cls.needs_orbit or cls.catalog or cls.supports_oscillator or cls.fades_at_horizon)
    assert local.allows_wideband and not ground.allows_wideband
    assert check_platform_params("ground", None) == {}
    with pytest.raises(ValueError, match="extra"):
        check_platform_params("ground", {"height_m": 3})


def test_unknown_platform_lists_the_known_ones() -> None:
    with pytest.raises(ValueError, match=r"unknown platform 'aircraft' \(known: satellite, ground, local\).*plugins:"):
        platform_class("aircraft")
    with pytest.raises(ValueError, match="unknown platform 'tethered'"):
        platform_class("tethered")  # on disk, but not declared by any scenario


def test_external_platforms_cannot_claim_core_only_capabilities() -> None:
    def make(**attrs: Any) -> Any:
        return type("Ext", (Platform,), {"__module__": "some_plugin", **attrs})

    for attrs in ({"level_mode": "link_budget"}, {"needs_orbit": True}, {"catalog": True}):
        with pytest.raises(TypeError, match="belong to the core satellite platform"):
            register_platform("ext")(make(**attrs))
    with pytest.raises(TypeError, match="a fixed platform has no passes"):
        register_platform("ext")(make(fixed=True, fades_at_horizon=True))
    with pytest.raises(TypeError, match="wideband signals need a fixed platform"):
        register_platform("ext")(make(allows_wideband=True))
    with pytest.raises(TypeError, match="Platform subclass"):
        register_platform("ext")(int)  # ty: ignore[invalid-argument-type]


def test_core_scenarios_are_unchanged_in_canonical_form() -> None:
    r = resolve(BASIC, now=0.0)
    doc = json.loads(r.canonical)
    assert all("platform_params" not in s for s in doc["sources"]) and "plugins" not in doc
    assert all(s.platform_params == {} for s in r.scenario.sources)


# --- the external platform ---


def test_tethered_scenario_resolves_with_defaults_and_capability_rules(tmp_path: Path) -> None:
    r = _resolved(tmp_path, _doc([_balloon()]))
    src = r.scenario.sources[0]
    assert src.platform == "tethered" and src.orbit is None and src.in_catalog is False
    assert src.platform_params["jitter_deg"] == 2.0  # the plugin's defaults are filled in
    assert json.loads(r.canonical)["sources"][0]["platform_params"]["swing_m"] == 50
    lv = src.emitters[0].level
    assert lv is not None and lv.rx_dbm is not None and lv.p_tx_dbw is None  # a received level, like ground
    assert resolve_document(json.loads(r.canonical), ROOT).canonical == r.canonical

    with pytest.raises(ValueError, match="unknown platform 'tethered'"):
        _resolved(tmp_path, _doc([_balloon()], declare=False))
    with pytest.raises(ValueError, match="height_m"):
        _resolved(tmp_path, _doc([_balloon(platform_params={"height_m": 1})]))
    with pytest.raises(ValueError, match="a tethered source has no orbit and is not in the catalog"):
        _resolved(tmp_path, _doc([_balloon(orbit={"norad": 57166})]))
    with pytest.raises(ValueError, match="a tethered source has no orbit and is not in the catalog"):
        _resolved(tmp_path, _doc([_balloon(in_catalog=True)]))
    osc = _balloon()
    osc["emitters"][0]["oscillator"] = {"pass_offset_sigma_hz": 50.0}
    with pytest.raises(ValueError, match="per-pass oscillators need a satellite"):
        _resolved(tmp_path, _doc([osc]))
    link = _balloon()
    link["emitters"][0]["level"] = {"p_tx_dbw": 3.0}
    with pytest.raises(ConfigError, match="a tethered level is cn0_dbhz or rx_dbm"):
        _resolved(tmp_path, _doc([link]))
    wide = _balloon()
    wide["emitters"][0].update(signal={"kind": "noise_comb"}, schedule={"type": "poisson_events", "rate_per_hour": 60})
    wide["emitters"][0].pop("level")
    with pytest.raises(ValueError, match="wideband signals belong to a local source"):
        _resolved(tmp_path, _doc([wide]))


def test_tethered_jitter_is_keyed_per_source(tmp_path: Path) -> None:
    r = _resolved(tmp_path, _doc([_balloon("a"), _balloon("b")]))
    a, b = (s.platform for s in Scene(r.scenario).sources)
    assert a.kind == "tethered" and not a.fixed and a.norad is None and a.in_catalog is False
    assert (a.az_deg, a.el_deg, a.phase) != (b.az_deg, b.el_deg, b.phase)  # ty: ignore[unresolved-attribute]
    assert abs(a.az_deg - 200) <= 2 and abs(a.el_deg - 35) <= 2  # ty: ignore[unresolved-attribute]
    again = Scene(_resolved(tmp_path, _doc([_balloon("b"), _balloon("a")])).scenario).source("a").platform
    assert (again.az_deg, again.phase) == (a.az_deg, a.phase)  # ty: ignore[unresolved-attribute]
    other_seed = _doc([_balloon("a")])
    other_seed["seed"] = 10
    assert Scene(_resolved(tmp_path, other_seed).scenario).sources[0].platform.phase != a.phase  # ty: ignore[unresolved-attribute]


def test_tethered_truth_has_geometry_and_doppler(tmp_path: Path) -> None:
    scene = Scene(_resolved(tmp_path, _doc([_balloon()])).scenario)
    em = scene.sources[0].emitters[0]
    plat = scene.sources[0].platform
    t = parse_iso(T0) + np.arange(0, 20, 0.25)
    tr = em.truth(t, 0.25)
    assert tr.t.size == t.size  # always visible
    np.testing.assert_allclose(tr.el_deg, plat.el_deg)  # ty: ignore[unresolved-attribute]
    assert tr.range_m.min() == pytest.approx(2950, abs=1) and tr.range_m.max() == pytest.approx(3050, abs=1)
    v_max = 50 * 2 * np.pi / 5
    assert np.abs(tr.range_rate_m_s).max() == pytest.approx(v_max, rel=5e-3)  # sampled every 0.25 s
    np.testing.assert_allclose(tr.doppler_hz, -F_HZ * tr.range_rate_m_s / C)
    assert np.abs(tr.doppler_hz).max() == pytest.approx(F_HZ * v_max / C, rel=5e-3)  # about 29 Hz
    assert np.ptp(tr.rx_dbm) == 0  # a received level: the same whatever the range
    assert plat.max_speed_km_s() == pytest.approx(v_max / 1000)
    # The range rate is the derivative of the range, and the two time grids agree
    k = sample_at(parse_iso(T0), 1000) + np.arange(5000)
    g = plat.geometry_ms(k)
    np.testing.assert_allclose(np.gradient(g.range_m, 1e-3)[1:-1], g.range_rate_m_s[1:-1], atol=2e-3)
    np.testing.assert_allclose(plat.geometry(k / 1000.0).range_m, g.range_m, atol=1e-4)
    # Passes are whole UTC days whichever window asks
    (p,) = plat.passes(parse_iso(T0), parse_iso(T0) + 60)
    assert (p.aos, p.los, p.aos_key) == (1793750400.0, 1793836800.0, 1793750400)
    assert plat.passes(p.aos + 80000, p.aos + 80001) == [p]
    assert len(plat.passes(p.los - 1, p.los + 1)) == 2
    table = truth.compute(scene, parse_iso(T0), parse_iso(T0) + 3, 1.0, ("tethered",))
    assert len(table.rows) == 3 and table.rows[0]["norad"] == "" and table.rows[0]["in_catalog"] == "false"
    assert truth.compute(scene, parse_iso(T0), parse_iso(T0) + 3, 1.0, ("satellite",)).rows == []


def test_tethered_render_carries_the_doppler_and_is_window_independent(tmp_path: Path) -> None:
    r = _resolved(tmp_path, _doc([_balloon()]))
    scene = Scene(r.scenario)
    n0 = sample_at(parse_iso(T0), FS) + 123_456
    n = round(2.5 * FS)
    iq = Renderer(scene, noise=False).iq(n0, n)
    # Instantaneous frequency of the tone against the truth: mix the nominal frequency away, average over 10 ms
    j = n0 + np.arange(n, dtype=np.int64)
    base = iq * np.exp(-2j * np.pi * ((round(F_HZ) - 137_000_000) * (j % FS) / FS))
    blk = FS // 100
    z = base[: n // blk * blk].reshape(-1, blk).mean(axis=1)
    f_meas = np.angle(z[1:] * np.conj(z[:-1])) * 100 / (2 * np.pi)
    t_mid = (n0 + blk * np.arange(1, len(z))) / FS
    want = scene.sources[0].emitters[0].truth(t_mid, 0.01).doppler_hz
    assert np.abs(want).max() > 25
    np.testing.assert_allclose(f_meas[2:-2], want[2:-2], atol=0.2)
    rd = Renderer(scene)
    one = rd.u8(n0, n)
    cuts = [0, 1_000_001, 3_333_333, n]
    parts = [Renderer(Scene(r.scenario)).u8(n0 + a, b - a) for a, b in itertools.pairwise(cuts)]
    np.testing.assert_array_equal(np.concatenate(parts), one)
    pooled = list(workers.iter_u8(r.canonical, ROOT, n0, n, FS, 2, task_s=1))
    np.testing.assert_array_equal(np.concatenate(pooled), one)


def test_truth_cli_takes_registered_platforms(tmp_path: Path) -> None:
    doc = _doc([_balloon(), {"id": "g", "platform": "ground", "emitters": _balloon()["emitters"]}])
    sc = tmp_path / "s.yaml"
    sc.write_text(yaml.safe_dump(doc))
    out = tmp_path / "truth.csv"
    res = runner.invoke(app, ["truth", str(sc), "-o", str(out), "--to", "+3s", "--platforms", "tethered"])
    assert res.exit_code == 0, res.output
    rows = out.read_text().splitlines()
    assert len(rows) == 4 and all(",balloon," in x for x in rows[1:])
    res = runner.invoke(app, ["truth", str(sc), "-o", str(out), "--to", "+3s"])
    assert res.exit_code == 0 and len(out.read_text().splitlines()) == 7  # default: every platform
    bad = runner.invoke(app, ["truth", str(sc), "-o", str(out), "--platforms", "aircraft"])
    assert bad.exit_code != 0 and "satellite, ground, local, tethered" in bad.output
