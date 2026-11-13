"""External modules (PLAN.md R1): the registry, `plugins:` loading, and the example signal `pretty_gfsk`."""

import itertools
import json
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from assist import plugins, registry
from assist.backend import dsp
from assist.cli import app
from assist.config.loader import ConfigError, load_scenario
from assist.config.resolve import resolve, resolve_document
from assist.emitters.schedules import Event
from assist.engine import workers
from assist.engine.render import Renderer
from assist.engine.scene import Scene
from assist.rng import KeyedRng
from assist.signals.base import Keys, Signal, check_params, get_kind, make_signal, register_signal, signal_class
from assist.timebase import parse_iso, sample_at

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = "examples/plugins/pretty_gfsk.py"
BASIC = ROOT / "tests/fixtures/scenario_basic.yaml"
FS = 2_400_000
CORE = ["cw", "oqpsk", "oqpsk_burst", "sdpsk", "lora", "d8psk", "gfsk", "am", "noise_band", "noise_comb"]
runner = CliRunner()


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run from the repo root, and leave no plugin active for the tests that follow."""
    monkeypatch.chdir(ROOT)
    yield
    plugins.activate([])


def _plugin(tmp_path: Path, name: str, kinds: list[str], body: str = "", api: str | None = "1.0") -> Path:
    """A small plugin file registering continuous kinds `kinds`."""
    src = (
        "import numpy as np\nfrom assist import api\n\n"
        f'ASSIST_API = "{api}"\n\n\nclass P(api.SignalParams):\n    level: float = 1.0\n\n'
    ).replace('ASSIST_API = "None"\n', "")
    for k in kinds:
        src += textwrap.dedent(f'''
            @api.register_signal("{k}")
            class K_{k}(api.Signal):
                Params = P

                def bandwidth_hz(self):
                    return 1000.0

                def preferred_rate(self):
                    return 4800

                def render(self, n0, n, rate):
                    return np.ones(n, dtype=np.complex64)
            ''')
    p = tmp_path / f"{name}.py"
    p.write_text(src + body)
    return p


# --- registry ---


def test_core_kinds_are_registered_by_the_decorator() -> None:
    assert registry.signals.names() == CORE
    assert all(registry.signals.origin(k) == "core" and not registry.signals.is_external(k) for k in CORE)
    assert get_kind("noise_comb").mode == "wideband" and get_kind("lora").mode == "event"
    assert signal_class("cw").kind == "cw" and signal_class("cw").__name__ == "CW"
    assert "bank_clip" not in registry.signals


def test_unknown_kind_lists_the_known_ones_and_points_at_plugins() -> None:
    with pytest.raises(ValueError, match=r"unknown signal kind 'msk' \(known: cw, oqpsk, .*noise_comb\).*plugins:"):
        get_kind("msk")
    with pytest.raises(ValueError, match="unknown signal kind"):
        check_params("pretty_gfsk", {})  # installed on disk, but no scenario has declared it


def test_core_name_cannot_be_registered_twice() -> None:
    base: Any = signal_class("cw")
    other = type("Other", (base,), {"__module__": "assist.signals.fake"})
    with pytest.raises(registry.RegistryError, match="registered twice"):
        registry.signals.register("cw")(other)
    assert signal_class("cw").__name__ == "CW"


def test_register_signal_checks_the_class() -> None:
    with pytest.raises(TypeError, match="Signal subclass"):
        register_signal("x")(int)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="Params"):
        register_signal("x")(type("NoParams", (Signal,), {}))


# --- loading ---


def test_path_plugin_is_active_only_while_declared() -> None:
    out = plugins.activate([{"path": PLUGIN}], (ROOT,))
    assert out[0]["path"] == PLUGIN and len(out[0]["sha256"]) == 64 and out[0]["version"] == "0.1.0"
    assert registry.signals.names() == [*CORE, "pretty_gfsk"]
    assert registry.signals.origin("pretty_gfsk") == PLUGIN and registry.signals.is_external("pretty_gfsk")
    assert get_kind("pretty_gfsk").mode == "event"
    assert check_params("pretty_gfsk", {})["rb"] == 38400
    assert [p.label for p in plugins.active()] == [PLUGIN]
    cls = signal_class("pretty_gfsk")
    plugins.activate([])
    assert registry.signals.names() == CORE and plugins.active() == []
    plugins.activate([{"path": PLUGIN}], (ROOT,))
    assert signal_class("pretty_gfsk") is cls  # the same file is imported once


def test_only_and_prefix(tmp_path: Path) -> None:
    p = _plugin(tmp_path, "two", ["aa", "bb"])
    plugins.activate([{"path": str(p)}])
    assert registry.signals.names()[-2:] == ["aa", "bb"]
    plugins.activate([{"path": str(p), "only": ["bb"]}])
    assert registry.signals.names() == [*CORE, "bb"]
    plugins.activate([{"path": str(p), "only": ["aa"], "prefix": "lab"}])
    assert registry.signals.names() == [*CORE, "lab.aa"]
    assert signal_class("lab.aa").kind == "aa"
    with pytest.raises(plugins.PluginError, match=r"only: zz not registered by it \(it has: aa, bb\)"):
        plugins.activate([{"path": str(p), "only": ["zz"]}])


def test_name_clashes_are_errors_and_a_prefix_resolves_them(tmp_path: Path) -> None:
    core = _plugin(tmp_path, "shadow", ["cw"])
    with pytest.raises(plugins.PluginError, match=r"signal 'cw' is a core name.*prefix"):
        plugins.activate([{"path": str(core)}])
    assert registry.signals.names() == CORE  # a failed activation leaves no plugin active
    plugins.activate([{"path": str(core), "prefix": "mine"}])
    assert "mine.cw" in registry.signals and signal_class("cw").__name__ == "CW"
    a, b = _plugin(tmp_path, "a", ["dup"]), _plugin(tmp_path, "b", ["dup"])
    with pytest.raises(plugins.PluginError, match=rf"plugins {a} and {b} both register signal 'dup'"):
        plugins.activate([{"path": str(a)}, {"path": str(b)}])
    plugins.activate([{"path": str(a)}, {"path": str(b), "prefix": "b"}])
    assert registry.signals.names()[-2:] == ["dup", "b.dup"]


def test_bad_entries_and_failing_plugins(tmp_path: Path) -> None:
    with pytest.raises(plugins.PluginError, match="file not found"):
        plugins.activate([{"path": "nowhere/x.py"}], (ROOT, tmp_path))
    with pytest.raises(plugins.PluginError, match="exactly one of name, path, module"):
        plugins.activate([{"path": PLUGIN, "module": "x"}])
    with pytest.raises(plugins.PluginError, match="unknown key"):
        plugins.activate([{"path": PLUGIN, "as": "x"}])
    with pytest.raises(plugins.PluginError, match="must be a list"):
        plugins.activate({"path": PLUGIN})
    empty = tmp_path / "empty.py"
    empty.write_text("X = 1\n")
    with pytest.raises(plugins.PluginError, match="registers nothing"):
        plugins.activate([{"path": str(empty)}])
    broken = tmp_path / "broken.py"
    broken.write_text("import no_such_module_at_all\n")
    with pytest.raises(plugins.PluginError, match="failed to import: ModuleNotFoundError"):
        plugins.activate([{"path": str(broken)}])
    with pytest.raises(plugins.PluginError, match=r"no `assist\.plugins` entry point and no importable module"):
        plugins.activate(["no_such_plugin_name"])
    p = _plugin(tmp_path, "once", ["once"])
    with pytest.raises(plugins.PluginError, match="listed twice"):
        plugins.activate([{"path": str(p)}, {"path": str(p)}])


def test_module_and_name_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _plugin(tmp_path, "modplug_r1", ["modkind"], body='\n__version__ = "2.5"\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    out = plugins.activate([{"module": "modplug_r1"}])
    assert out == [{"module": "modplug_r1", "version": "2.5", "api": "1.0"}] and "modkind" in registry.signals
    plugins.activate([])
    assert "modkind" not in registry.signals
    out = plugins.activate(["modplug_r1"])  # a bare string: an entry point name, else a module
    assert out == [{"name": "modplug_r1", "version": "2.5", "api": "1.0"}] and "modkind" in registry.signals
    assert plugins.from_cli("psk8") == {"name": "psk8"} and plugins.from_cli("a/b.py") == {"path": "a/b.py"}


def test_file_is_found_in_cwd_first_then_next_to_the_scenario(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sc_dir, cwd = tmp_path / "sc", tmp_path / "cwd"
    (sc_dir / "plugins").mkdir(parents=True)
    (cwd / "plugins").mkdir(parents=True)
    near = _plugin(sc_dir / "plugins", "k", ["near"])
    assert plugins.find_file({"path": "plugins/k.py"}, (cwd, sc_dir)) == near.resolve()
    here = _plugin(cwd / "plugins", "k", ["here"])
    assert plugins.find_file({"path": "plugins/k.py"}, (cwd, sc_dir)) == here.resolve()
    assert plugins.find_file({"path": str(near)}, ()) == near.resolve()  # absolute paths are taken as they are


# --- the API version ---


def test_api_is_frozen() -> None:
    from assist import api

    assert api.API_VERSION == plugins.API_VERSION == "1.0"
    assert api.__all__ == [
        "API_VERSION",
        "Event",
        "Geometry",
        "Keys",
        "OutFile",
        "Pass",
        "Platform",
        "PlatformContext",
        "PlatformParams",
        "RenderAhead",
        "RngScope",
        "ServingParams",
        "ServingSink",
        "ShapedParams",
        "ShapedSignal",
        "Signal",
        "SignalParams",
        "Sink",
        "SinkContext",
        "SinkParams",
        "StreamServer",
        "StreamSink",
        "float_range",
        "int_range",
        "register_platform",
        "register_signal",
        "register_sink",
        "rrc_taps",
        "unpack",
    ]  # a name added here raises the minor of API_VERSION; one removed or changed, the major
    assert all(hasattr(api, name) for name in api.__all__)
    assert callable(dsp.random_bytes) and callable(dsp.white)  # the two kernels that are part of the API


def test_plugin_must_declare_a_compatible_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    none = _plugin(tmp_path, "noapi", ["noapi"], api=None)
    with pytest.raises(plugins.PluginError, match=rf'plugin {none}: the module does not say .*ASSIST_API = "1.0"'):
        plugins.activate([{"path": str(none)}])
    assert registry.signals.names() == CORE
    major = _plugin(tmp_path, "major", ["major"], api="2.0")
    with pytest.raises(
        plugins.PluginError, match=r"written for assist.api 2.0; this is assist.api 1.0 \(another major"
    ):
        plugins.activate([{"path": str(major)}])
    newer = _plugin(tmp_path, "newer", ["newer"], api="1.7")
    with pytest.raises(plugins.PluginError, match=r"written for assist.api 1.7; this is the older assist.api 1.0"):
        plugins.activate([{"path": str(newer)}])
    for bad in ("1", "1.0.0", "one", ">=1.0"):
        odd = _plugin(tmp_path, f"odd{len(bad)}{bad[:1].isdigit()}", ["odd"], api=bad)
        with pytest.raises(plugins.PluginError, match=r'must be a string "major\.minor"'):
            plugins.activate([{"path": str(odd)}])
    monkeypatch.setattr(plugins, "API_VERSION", "1.3")  # a later minor of this package still takes 1.0 and 1.3
    older = _plugin(tmp_path, "older", ["older"])
    same = _plugin(tmp_path, "same", ["same"], api="1.3")
    out = plugins.activate([{"path": str(older)}, {"path": str(same)}])
    assert [e["api"] for e in out] == ["1.0", "1.3"] and [p.api for p in plugins.active()] == ["1.0", "1.3"]
    with pytest.raises(plugins.PluginError, match=r"the older assist\.api 1\.3"):
        plugins.activate([{"path": str(_plugin(tmp_path, "ahead", ["ahead"], api="1.4"))}])


def test_named_plugin_from_an_entry_point(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A `name:` entry is looked up in the entry-point group `assist.plugins` (faked here: nothing is installed)."""
    import importlib.metadata as md

    _plugin(tmp_path, "eppkg_r4", ["epkind"])
    monkeypatch.syspath_prepend(str(tmp_path))

    class Dist:
        name, version = "assist-eppkg", "3.1"

    ep = md.EntryPoint("epkind_plugin", "eppkg_r4", plugins.ENTRY_POINT_GROUP)
    vars(ep).update(dist=Dist())
    monkeypatch.setattr(plugins, "_entry_points", lambda name: [ep] if name == "epkind_plugin" else [])
    out = plugins.activate(["epkind_plugin"])
    assert out == [{"name": "epkind_plugin", "version": "3.1", "api": "1.0"}]
    (p,) = plugins.active()
    assert p.distribution == "assist-eppkg" and p.registered == (("signal", "epkind"),) and p.sha256 is None
    monkeypatch.setattr(plugins, "_entry_points", lambda name: [ep, ep])
    with pytest.raises(plugins.PluginError, match="several installed packages provide this entry point"):
        plugins.activate(["epkind_plugin"])


def test_plugins_list_command(tmp_path: Path) -> None:
    res = runner.invoke(app, ["plugins", "list"])
    assert res.exit_code == 0, res.output
    lines = res.output.splitlines()
    assert lines[0] == "signals" and "platforms" in lines and "sinks" in lines
    assert any(x.split() == ["lora", "event", "core"] for x in lines)
    assert any(x.split() == ["satellite", "geometry", "core"] for x in lines)
    assert any(x.split() == ["rtl_tcp", "serving", "core"] for x in lines)
    assert lines[-2:] == ["plugins (assist.api 1.0)", "  none"] and "pretty_gfsk" not in res.output
    res = runner.invoke(
        app, ["plugins", "list", "scenarios/pretty_gfsk.yaml", "--plugin", "examples/plugins/tcp_sink.py"]
    )
    assert res.exit_code == 0, res.output
    lines = res.output.splitlines()
    assert any(x.split() == ["pretty_gfsk", "event", PLUGIN] for x in lines)
    assert any(x.split() == ["tcp", "serving", "examples/plugins/tcp_sink.py"] for x in lines)
    i = lines.index("plugins (assist.api 1.0)")
    assert lines[i + 1].startswith(f"  {PLUGIN}  (version 0.1.0, api 1.0, sha256 ")
    assert lines[i + 2] == f"    path: {ROOT / PLUGIN}" and lines[i + 3] == "    registers: signal pretty_gfsk"
    assert "registers: sink tcp" in lines[-1]
    res = runner.invoke(app, ["plugins", "list", str(BASIC)])
    assert res.exit_code == 0 and res.output.splitlines()[-1] == "  none listed"
    # A scenario whose plugin cannot be loaded: the error names the entry, and a broken rest does not matter
    sc = tmp_path / "s.yaml"
    sc.write_text("plugins: [{path: nowhere/x.py}]\nsources: [{id: a, platform: nonsense}]\n")
    res = runner.invoke(app, ["plugins", "list", str(sc)])
    assert res.exit_code == 2 and "plugin nowhere/x.py: file not found" in res.output
    sc.write_text(f"plugins: [{{path: {ROOT / PLUGIN}, prefix: lab}}]\nsources: [{{id: a, platform: nonsense}}]\n")
    res = runner.invoke(app, ["plugins", "list", str(sc)])
    assert res.exit_code == 0 and "registers: signal lab.pretty_gfsk" in res.output


# --- scenarios ---


def _doc(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "seed": 4,
        "clock": {"start": "2026-11-04T12:00:00Z", "duration": "10min"},
        "station": {"file": None, "lat": 55.5, "lon": 37.1, "alt_m": 172},
        "plugins": [{"path": PLUGIN}],
        "sources": [
            {
                "id": "pretty",
                "orbit": {"synthetic_pass": {"aos": "-3min", "max_el_deg": 50, "direction": "desc", "side": "east"}},
                "emitters": [
                    {
                        "id": "tm",
                        "signal": {"kind": "pretty_gfsk", "params": {"burst_ms": [150, 300]}},
                        "freq_hz": 137.78e6,
                        "schedule": {"type": "exp_gaps", "mean_gap_s": 0.6, "min_gap_s": 0.4},
                        "level": {"cn0_zenith_dbhz": 70},
                    }
                ],
            }
        ],
        **(extra or {}),
    }


def _load(tmp_path: Path, doc: dict[str, Any]) -> dict[str, Any]:
    """The document as a scenario file would give it (band defaults and presets applied)."""
    p = tmp_path / "scenario.yaml"
    p.write_text(yaml.safe_dump(doc))
    return load_scenario(p)


def test_scenario_declares_its_plugins(tmp_path: Path) -> None:
    r = resolve_document(_load(tmp_path, _doc()), ROOT)
    (p,) = r.scenario.plugins
    assert p.path == PLUGIN and p.version == "0.1.0" and p.sha256 is not None
    assert json.loads(r.canonical)["plugins"] == [
        {"api": "1.0", "path": PLUGIN, "sha256": p.sha256, "version": "0.1.0"}
    ]
    assert r.scenario.sources[0].emitters[0].signal.params["rb"] == 38400  # the plugin's defaults are filled in
    again = resolve_document(json.loads(r.canonical), ROOT)  # resolving a resolved scenario changes nothing
    assert again.canonical == r.canonical and again.hash == r.hash
    undeclared = _doc()
    del undeclared["plugins"]
    with pytest.raises(ValueError, match=r"unknown signal kind 'pretty_gfsk'.*plugins:"):
        resolve_document(_load(tmp_path, undeclared), ROOT)
    typo = _doc()
    typo["sources"][0]["emitters"][0]["signal"]["params"] = {"rolloff": 1}
    with pytest.raises(ValueError, match="rolloff"):
        resolve_document(_load(tmp_path, typo), ROOT)


def test_core_only_scenarios_keep_their_hash() -> None:
    r = resolve(BASIC, now=0.0)
    assert r.scenario.plugins == [] and "plugins" not in json.loads(r.canonical)
    assert registry.signals.names() == CORE


def test_changed_plugin_file_is_refused(tmp_path: Path) -> None:
    p = _plugin(tmp_path, "chg", ["chg"])
    (entry,) = plugins.activate([{"path": str(p)}])
    p.write_text(p.read_text() + "\n# edited\n")
    with pytest.raises(plugins.PluginError, match="has changed since the scenario was resolved"):
        plugins.activate([entry])
    assert plugins.activate([{"path": str(p)}])[0]["sha256"] != entry["sha256"]  # resolving again picks it up


def test_scenario_file_finds_a_plugin_next_to_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "plugins").mkdir()
    _plugin(tmp_path / "plugins", "local_r1", ["local_kind"])
    doc = _doc({"plugins": [{"path": "plugins/local_r1.py"}]})
    doc["sources"][0]["emitters"][0].update(signal={"kind": "local_kind"}, schedule={"type": "continuous"})
    sc = tmp_path / "s.yaml"
    sc.write_text(yaml.safe_dump(doc))
    r = resolve(sc)  # cwd is the repo root: the file is found next to the scenario
    assert r.scenario.plugins[0].path == "plugins/local_r1.py"
    Scene(r.scenario)  # and again when the scene is built, from where it was found
    with pytest.raises(ConfigError, match="file not found"):
        resolve_document(load_scenario(sc), ROOT)  # no scenario directory given


def test_cli_plugin_option_and_errors(tmp_path: Path) -> None:
    doc = _doc()
    del doc["plugins"]
    sc = tmp_path / "s.yaml"
    sc.write_text(yaml.safe_dump(doc))
    res = runner.invoke(app, ["validate", str(sc)])
    assert res.exit_code == 2 and "unknown signal kind 'pretty_gfsk'" in res.output
    res = runner.invoke(app, ["validate", str(sc), "--plugin", PLUGIN])
    assert res.exit_code == 0, res.output
    res = runner.invoke(app, ["resolve", str(sc), "--plugin", PLUGIN])
    assert json.loads(res.stdout)["plugins"][0]["path"] == PLUGIN  # recorded in the resolved scenario
    res = runner.invoke(app, ["validate", str(sc), "--plugin", "missing/x.py"])
    assert res.exit_code == 2 and "file not found" in res.output


# --- the example signal ---


def _signal(params: dict[str, Any] | None = None, eid: str = "tm") -> Signal:
    plugins.activate([{"path": PLUGIN}], (ROOT,))
    return make_signal("pretty_gfsk", params or {}, Keys(KeyedRng(7), "pretty", eid), dsp)


def test_external_signal_keys_are_namespaced() -> None:
    sig = _signal()
    assert sig.keys.prefix == ("signal", "pretty_gfsk", "pretty", "tm")
    assert sig.keys.key64("data") == KeyedRng(7).key64("signal", "pretty_gfsk", "pretty", "tm", "data")
    core = make_signal("gfsk", {}, Keys(KeyedRng(7), "pretty", "tm"), dsp)
    assert core.keys.prefix == ("pretty", "tm")  # core kinds keep their keys


def test_pretty_gfsk_is_window_independent_with_unit_power() -> None:
    sig = _signal()
    rate = sig.preferred_rate()
    assert rate == 4 * 38400 and sig.rate_step() == 38400 and sig.bandwidth_hz() == pytest.approx(57600)
    evs = [Event(1000.00001, 1000.2, {"bits": 7680}), Event(1000.3, 1000.5, {"bits": 5000})]
    n0 = 1000 * rate - 500
    n = round(0.6 * rate)
    whole = sig.render_events(evs, n0, n, rate)
    fresh = _signal()
    cuts = [0, 137, 20_000, 20_001, 61_000, n]
    parts = [fresh.render_events(evs, n0 + a, b - a, rate) for a, b in itertools.pairwise(cuts)]
    np.testing.assert_array_equal(np.concatenate(parts), whole)
    # First sample of the first burst: on the bit grid, after the event start
    a0 = sig.first_bit(evs[0]) * 4 - n0  # ty: ignore[unresolved-attribute]
    assert a0 == 504 and whole[:a0].max() == 0 and abs(whole[a0]) > 0
    body = whole[a0 + 8 * 4 : a0 + (7680 - 8) * 4]  # between the ramps: constant envelope, unit power
    np.testing.assert_allclose(np.abs(body), 1.0, atol=1e-6)
    assert np.abs(whole[a0 + 7680 * 4 : a0 + 7680 * 4 + 100]).max() == 0  # off between the bursts
    # Instantaneous frequency: +-h rb / 2 = +-9600 Hz, alternating through the preamble
    f = np.angle(body[1:] * np.conj(body[:-1])) * rate / (2 * np.pi)
    assert np.abs(f).max() == pytest.approx(9600, rel=0.02)
    pre = f[1 : (60 - 8) * 4 : 4]  # bit centres of preamble bits 8..59 (body starts at bit 8)
    assert (np.sign(pre[1:]) == -np.sign(pre[:-1])).all()
    other = _signal(eid="tm2").render_events(evs, n0, n, rate)
    assert not np.array_equal(other, whole)  # another emitter: other data


def test_pretty_gfsk_scenario_renders_the_same_on_workers(tmp_path: Path) -> None:
    r = resolve_document(_load(tmp_path, _doc()), ROOT)
    n0 = sample_at(parse_iso("2026-11-04T12:00:00Z"), FS) + 333_333
    n = round(2.5 * FS)
    rd = Renderer(Scene(r.scenario))
    one = rd.u8(n0, n)
    bare = Renderer(Scene(r.scenario), noise=False).iq(n0, n)
    assert np.abs(bare).max() > 0  # bursts are in the window
    parts = list(workers.iter_u8(r.canonical, ROOT, n0, n, FS, 2, task_s=1))
    np.testing.assert_array_equal(np.concatenate(parts), one)
    assert registry.signals.names() == [*CORE, "pretty_gfsk"]
