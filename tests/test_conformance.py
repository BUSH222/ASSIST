"""`assist.testing`: the conformance checks pass for the core modules and the example plugins, and catch modules that
break the rules of `assist.api`."""

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from assist import plugins, registry, testing

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = "examples/plugins"

# Arguments the core kinds need: a required parameter, longer events, or a level that is not "unit while on".
SIGNAL_ARGS: dict[str, dict[str, Any]] = {
    "noise_band": {"params": {"width_hz": 20_000}},
    "am": {"seconds": 8.0, "event_s": 3.0},
    "oqpsk_burst": {"power_tol_db": 6.0},  # bursts with gaps inside a continuous signal
    "lora": {"atol": 1e-6},  # chirp times are formed from the window start: 1 ulp of float32 between cuts
}


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.chdir(ROOT)
    yield
    plugins.activate([])


def _bad(tmp_path: Path, name: str, body: str) -> Any:
    p = tmp_path / f"{name}.py"
    p.write_text('import numpy as np\nfrom assist import api\n\nASSIST_API = "1.0"\n' + textwrap.dedent(body))
    return testing.load_plugin(str(p))


# --- the examples, as external modules ---


def _guide_plugin(tmp_path: Path, name: str) -> Any:
    """The complete plugin PLUGINS.md prints for `name` (its fenced python block that opens with that docstring)."""
    blocks = (ROOT / "PLUGINS.md").read_text().split("```python\n")[1:]
    (code,) = [b.split("```")[0] for b in blocks if b.startswith(f'"""`{name}`')]
    p = tmp_path / f"{name}.py"
    p.write_text(code)
    return testing.load_plugin(str(p))


def test_guide_signal_conforms(tmp_path: Path) -> None:
    mod = _guide_plugin(tmp_path, "fsk2")
    testing.check_signal(mod.Fsk2)
    testing.check_signal("fsk2", {"rb": 4800, "sps": 5}, seed=3)


def test_guide_sink_conforms(tmp_path: Path) -> None:
    mod = _guide_plugin(tmp_path, "power_log")
    testing.check_sink(mod.PowerLog, seconds=3.0)


def test_example_signal_conforms() -> None:
    mod = testing.load_plugin(f"{EXAMPLES}/pretty_gfsk.py")
    testing.check_signal(mod.PrettyGfsk)
    testing.check_signal("pretty_gfsk", {"burst_ms": [150, 300], "bt": 0.3}, seed=5)


def test_example_platform_conforms() -> None:
    mod = testing.load_plugin(f"{EXAMPLES}/jitter_platform.py")
    testing.check_platform(mod.Tethered)
    testing.check_platform("tethered", {"range_m": 3000, "swing_m": 50, "period_ms": 5000}, seed=3)


@pytest.mark.parametrize("fmt", ["cu8", "cs16", "cf32"])
def test_example_sink_conforms(fmt: str) -> None:
    mod = testing.load_plugin(f"{EXAMPLES}/tcp_sink.py")
    testing.check_sink(mod.TcpSink, {"format": fmt}, seconds=0.25)


def test_load_plugin_makes_it_the_active_one() -> None:
    mod = testing.load_plugin({"path": f"{EXAMPLES}/pretty_gfsk.py", "prefix": "lab"})
    assert registry.signals.names()[-1] == "lab.pretty_gfsk" and plugins.active()[0].module is mod
    testing.check_signal("lab.pretty_gfsk")
    with pytest.raises(testing.ConformanceError, match="unknown signal kind 'pretty_gfsk'"):
        testing.check_signal("pretty_gfsk")
    with pytest.raises(testing.ConformanceError, match="is not a Signal subclass"):
        testing.check_signal(int)  # ty: ignore[invalid-argument-type]


# --- the core modules keep the same rules ---


def test_core_signals_conform() -> None:
    for kind in registry.signals.core_names():
        testing.check_signal(kind, **SIGNAL_ARGS.get(kind, {}))


def test_core_fixed_platforms_and_sinks_conform() -> None:
    testing.check_platform("ground")
    testing.check_platform("local")
    for name in registry.sinks.core_names():
        testing.check_sink(name, seconds=1.0 if name == "fft" else 0.25)  # a waterfall row is about 1 s


# --- what the checks catch ---

SIGNAL = """
class P({params_base}):
    {params_extra}
    amp: float = 1.0


@api.register_signal("{name}")
class S(api.Signal):
    Params = P

    def __init__(self, params, keys, dsp):
        super().__init__(params, keys, dsp)
        self.calls = 0

    def bandwidth_hz(self):
        return 1000.0

    def preferred_rate(self):
        return {rate}

    def render(self, n0, n, rate):
        self.calls += 1
        {render}
"""


def _signal(tmp_path: Path, name: str, **kw: str) -> Any:
    args = {
        "name": name,
        "params_base": "api.SignalParams",
        "params_extra": "",
        "rate": "4800",
        "render": "return np.ones(n, dtype=np.complex64)",
        **kw,
    }
    return _bad(tmp_path, name, SIGNAL.format(**args)).S


def test_signal_faults_are_reported(tmp_path: Path) -> None:
    testing.check_signal(_signal(tmp_path, "fine"))
    # Content addressed by the index inside the window, not by the absolute index
    rel = _signal(tmp_path, "relative", render="return np.exp(0.01j * np.arange(n)).astype(np.complex64)")
    with pytest.raises(testing.ConformanceError, match="in pieces differs from rendering it whole"):
        testing.check_signal(rel)
    # Randomness that is not keyed
    rnd = "return np.exp(2j * np.pi * np.random.default_rng().random(n)).astype(np.complex64)"
    with pytest.raises(testing.ConformanceError, match="in pieces differs"):
        testing.check_signal(_signal(tmp_path, "unkeyed", render=rnd))
    with pytest.raises(testing.ConformanceError, match=r"mean power while on is \+6\.02 dB"):
        testing.check_signal(_signal(tmp_path, "loud", render="return np.full(n, 2, dtype=np.complex64)"))
    with pytest.raises(testing.ConformanceError, match=r"must return complex64 of shape"):
        testing.check_signal(_signal(tmp_path, "wide", render="return np.ones(n)"))
    with pytest.raises(testing.ConformanceError, match="not a multiple of rate_step"):
        testing.check_signal(_signal(tmp_path, "oddrate", rate="4850"))
    loose = _signal(tmp_path, "loose", params_extra='model_config = {"extra": "ignore"}')
    with pytest.raises(testing.ConformanceError, match="unknown parameter is accepted"):
        testing.check_signal(loose)
    testing.load_plugin(str(tmp_path / "fine.py"))
    with pytest.raises(testing.ConformanceError, match="the given parameters are not valid"):
        testing.check_signal("fine", {"nope": 1})


def test_platform_faults_are_reported(tmp_path: Path) -> None:
    body = '''
    class Drift(api.Platform):
        """Passes that start at the window asked for; a range that does not match its rate."""

        def __init__(self, ctx):
            self.source_id = self.name = ctx.source_id

        @classmethod
        def build(cls, ctx):
            return cls(ctx)

        def geometry(self, t):
            t = np.atleast_1d(np.asarray(t, dtype=np.float64))
            one = np.ones(t.shape)
            return api.Geometry(30 * one, 90 * one, 1e6 + self.rate * (t - 1.7937936e9), self.speed * one, 0 * one)

        def passes(self, t0, t1):
            a = t0 if self.window else 0.0
            return [api.Pass(0, self.name, a, a + 1, 4e9, 30.0, int(a))]

        def max_speed_km_s(self):
            return self.top


    @api.register_platform("windowed")
    class Windowed(Drift):
        rate, speed, top, window = 10.0, 10.0, 0.01, True


    @api.register_platform("slow")
    class Slow(Drift):
        rate, speed, top, window = 10.0, 10.0, 0.001, False


    @api.register_platform("inconsistent")
    class Inconsistent(Drift):
        rate, speed, top, window = 10.0, 500.0, 1.0, False


    @api.register_platform("sound")
    class Sound(Drift):
        rate, speed, top, window = 10.0, 10.0, 0.01, False
    '''
    mod = _bad(tmp_path, "platforms_bad", body)
    testing.check_platform(mod.Sound)
    with pytest.raises(testing.ConformanceError, match="a pass depends on the window asked for"):
        testing.check_platform(mod.Windowed)
    with pytest.raises(testing.ConformanceError, match=r"max_speed_km_s\(\) = 0.001 is below the range rate"):
        testing.check_platform(mod.Slow)
    with pytest.raises(testing.ConformanceError, match="range_rate_m_s is not the time derivative of range_m"):
        testing.check_platform(mod.Inconsistent)
    with pytest.raises(testing.ConformanceError, match="the given parameters are not valid"):
        testing.check_platform(mod.Sound, {"height_m": 1})


def test_sink_faults_are_reported(tmp_path: Path) -> None:
    body = '''
    from pathlib import Path


    class Base(api.Sink):
        """Writes one file; subclasses break one rule each."""

        marks = False
        listed = True
        obj = False

        def __init__(self, params, ctx):
            super().__init__(params, ctx)
            self.path = ctx.out / "x.bin"
            self.f = None
            self.n = 0

        def files(self):
            return [api.OutFile(self.path, 0, self.planned(), self.ctx.n0)] if self.listed else []

        def open(self):
            if self.listed:
                self.f = self.path.open("wb")

        def write(self, u8):
            if self.f is not None:
                self.f.write(b"|" if self.marks else b"")
                self.f.write(u8.tobytes())
            self.n += len(u8) // 2

        def close(self):
            if self.f is not None:
                self.f.close()

        abort = close

        def info(self):
            return {"samples": self.n, "where": self.path if self.obj else str(self.path)}


    @api.register_sink("plain")
    class Plain(Base):
        pass


    @api.register_sink("marks")
    class Marks(Base):
        marks = True


    @api.register_sink("opaque")
    class Opaque(Base):
        obj = True


    @api.register_sink("greedy")
    class Greedy(Base):
        def planned(self):
            return self.ctx.n + 1
    '''
    mod = _bad(tmp_path, "sinks_bad", body)
    testing.check_sink(mod.Plain, seconds=0.1, out=tmp_path / "kept")
    assert (tmp_path / "kept" / "run0" / "x.bin").stat().st_size == 2 * 240_000  # `out` keeps what was written
    with pytest.raises(testing.ConformanceError, match="differ when the same stream arrives in pieces"):
        testing.check_sink(mod.Marks, seconds=0.1)
    with pytest.raises(testing.ConformanceError, match="JSON-serialisable"):
        testing.check_sink(mod.Opaque, seconds=0.1)
    with pytest.raises(testing.ConformanceError, match=r"planned\(\) must be in 1 \.\. 240000"):
        testing.check_sink(mod.Greedy, seconds=0.1)
    with pytest.raises(testing.ConformanceError, match="the given parameters are not valid"):
        testing.check_sink(mod.Plain, {"colour": "red"})
