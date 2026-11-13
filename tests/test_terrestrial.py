"""M7, the terrestrial environment (plan §5, §6.5, §10.1): ground/local platforms, per-event draws, d8psk, am,
noise_band, noise_comb, and the band's terrestrial scenario. No samples needed (flat noise shape: no profile, except
the band-file checks, which use the committed profile).

- Signals: VDL bursts (symbol grid, sync sequence, Gray-coded steps, ramps, unit power, bandwidth), AM (carrier,
  voice band, PTT ramps, pauses), noise_band (power, width), noise_comb (comb spacing, level, ripple).
- Engine: ground levels resolve to rx_dbm; each event's offset and level in truth and in the rendered signal; C/N0
  from the uint8 stream; two AM transmitters on one channel beat; T-WIN, worker processes, T-KEY.
- Schedules: event ramps, the dead time (min_gap_s), the bounded slot cache.
- Truth: ground rows without geometry, every burst with its frequency and strength, the platform filter.
- bands/vhf137/terrestrial_default.yaml is what the profile derives, and it renders.
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from pydantic import ValidationError
from scipy.signal import welch
from typer.testing import CliRunner

from assist import measure
from assist import truth as T
from assist.backend import dsp
from assist.cli import app
from assist.config.loader import ConfigError, band_dir, load_scenario
from assist.config.resolve import resolve_document, t_sys_of
from assist.emitters import link
from assist.emitters.schedules import SLOT_CACHE, DurationDist, Event, PoissonEvents, event_envelope
from assist.engine import workers
from assist.engine.render import Renderer
from assist.engine.scene import Scene
from assist.orbits import catalog as cat
from assist.profile import terrestrial as ter
from assist.rng import KeyedRng
from assist.signals.am import AM
from assist.signals.base import Keys, make_signal
from assist.signals.d8psk import D8PSK, GRAY, SYNC, D8pskParams, rc_taps
from assist.signals.gfsk import GFSK
from assist.signals.noise import beta_of
from assist.timebase import parse_iso, sample_at

ROOT = Path(__file__).resolve().parent.parent
FS = 2_400_000
FC = 137.0e6
T0 = parse_iso("2026-11-04T12:00:00Z")
N0 = sample_at(T0, FS)
STATION = {"file": None, "lat": 55.5, "lon": 37.1, "alt_m": 172}

SOURCES: list[dict[str, Any]] = [
    {"id": "gs", "preset": "vdl_ground", "freq_hz": 136.925e6, "schedule": {"rate_per_hour": 3600}},
    {"id": "air", "preset": "vdl_aircraft", "freq_hz": 136.975e6, "schedule": {"rate_per_hour": 7200}},
    {"id": "am", "preset": "atc_am", "freq_hz": 136.2e6, "schedule": {"rate_per_hour": 600}},
    {"id": "c138", "preset": "carrier_138", "schedule": {"rate_per_hour": 720}},
    {
        "id": "pe",
        "platform": "ground",
        "emitters": [
            {
                "id": "beep",
                "signal": {"kind": "cw"},
                "freq_hz": 137.3e6,
                "schedule": {
                    "type": "poisson_events",
                    "rate_per_hour": 360,
                    "duration": {"dist": "fixed", "s": 0.4},
                    "ramp_ms": 5,
                    "min_gap_s": 0.2,
                },
                "level": {"cn0_dbhz": 60},
                "per_event": {
                    "offset_hz": [-1000, 1000],
                    "level_db": {"dist": "normal", "sigma_db": 3, "min_db": -5, "max_db": 5},
                },
            }
        ],
    },
    {
        "id": "tone",
        "platform": "ground",
        "emitters": [{"id": "cw", "signal": {"kind": "cw"}, "freq_hz": 137.4e6, "level": {"cn0_dbhz": 45}}],
    },
    {
        "id": "bird",
        "preset": "birdies",
        "emitters": [
            {"id": "b1", "signal": {"kind": "cw"}, "freq_hz": 137.6285e6, "level": {"cn0_dbhz": 30}},
            {
                "id": "b2",
                "signal": {"kind": "noise_band", "params": {"width_hz": 2200}},
                "freq_hz": 136.05e6,
                "level": {"cn0_dbhz": 36},
            },
        ],
    },
    {"id": "hum", "preset": "usb_humps", "schedule": {"rate_per_hour": 4, "duration": {"dist": "fixed", "s": 400}}},
]


def _resolve(tmp: Path, sources: list[dict[str, Any]], name: str = "ter", seed: int = 5):
    doc = {"seed": seed, "clock": {"start": "2026-11-04T12:00:00Z", "duration": "10min"}, "station": STATION}
    doc["sources"] = sources
    p = tmp / f"{name}.yaml"
    p.write_text(yaml.safe_dump(doc))
    return resolve_document(load_scenario(p), ROOT)


@pytest.fixture(scope="module")
def resolved(tmp_path_factory: pytest.TempPathFactory):
    return _resolve(tmp_path_factory.mktemp("ter"), SOURCES)


@pytest.fixture(scope="module")
def scene(resolved) -> Scene:
    return Scene(resolved.scenario)


def _em(scene: Scene, sid: str):
    return scene.source(sid).emitters[0]


def _keys(eid: str = "e") -> Keys:
    return Keys(KeyedRng(1), "s", eid)


# --- schema and resolve -------------------------------------------------------------------------------------------


def _cw_ground(**kw: Any) -> dict[str, Any]:
    em = {"id": "c", "signal": {"kind": "cw"}, "freq_hz": 137.3e6, "level": {"cn0_dbhz": 40}}
    em.update(kw.pop("em", {}))
    return {"id": "g", "platform": "ground", "emitters": [em], **kw}


@pytest.mark.parametrize(
    ("src", "msg"),
    [
        (_cw_ground(orbit={"norad": 41188}), "no orbit"),
        (
            {
                "id": "s",
                "emitters": [{"id": "c", "signal": {"kind": "cw"}, "freq_hz": 137e6, "level": {"p_tx_dbw": 1}}],
            },
            "needs an orbit",
        ),
        (_cw_ground(em={"level": {"cal": "orbcomm"}}), "cn0_dbhz or rx_dbm"),
        (_cw_ground(em={"oscillator": {"drift_hz_s": 0.2}}), "need a satellite"),
        (_cw_ground(em={"per_event": {"offset_hz": 100}}), "per_event needs an event schedule"),
        (
            _cw_ground(
                em={
                    "signal": {"kind": "noise_comb"},
                    "level": None,
                    "schedule": {"type": "poisson_events", "rate_per_hour": 1},
                }
            ),
            "local source",
        ),
        (
            {
                **_cw_ground(),
                "platform": "local",
                "emitters": [
                    {
                        "id": "n",
                        "signal": {"kind": "noise_comb"},
                        "freq_hz": 137e6,
                        "level": {"cn0_dbhz": 3},
                        "schedule": {"type": "poisson_events", "rate_per_hour": 1},
                    }
                ],
            },
            "takes no level",
        ),
    ],
)
def test_schema_rules(tmp_path: Path, src: dict[str, Any], msg: str) -> None:
    with pytest.raises((ValidationError, ConfigError), match=msg):
        _resolve(tmp_path, [src])


def test_satellites_refuse_fixed_levels(tmp_path: Path) -> None:
    src = {
        "id": "z",
        "preset": "zombie",
        "orbit": {"synthetic_pass": {"aos": "+1min", "max_el_deg": 40}},
        "freq_hz": 137.3e6,
        "level": {"cn0_dbhz": 40},
    }
    with pytest.raises(ConfigError, match="ground and local"):
        _resolve(tmp_path, [src])


def test_ground_levels_resolve(resolved) -> None:
    sc = resolved.scenario
    t_sys = t_sys_of(sc.frontend)
    gs = next(s for s in sc.sources if s.id == "gs")
    lv = gs.emitters[0].level
    assert gs.platform == "ground" and gs.orbit is None and gs.in_catalog is False
    assert lv is not None and lv.p_tx_dbw is None
    assert lv.rx_dbm == pytest.approx(55 + 30 + link.kt_dbw_hz(t_sys), abs=1e-9)
    assert "cn0_dbhz 55" in str(lv.resolved_from)
    hum = next(s for s in sc.sources if s.id == "hum")
    assert hum.platform == "local" and hum.emitters[0].level is None
    assert resolve_document(sc.model_dump(mode="json"), ROOT).canonical == resolved.canonical
    assert cat.entries(sc, ROOT, sources_only=True) == []  # ground and local sources are not in the catalog


# --- d8psk --------------------------------------------------------------------------------------------------------


def _burst(sig: D8PSK, start: float, n_sym: int) -> Event:
    return Event(start, start + n_sym / 10_500, {"symbols": n_sym, "duration_s": n_sym / 10_500})


def test_d8psk_burst_structure() -> None:
    sig = make_signal("d8psk", {}, _keys(), dsp)
    assert isinstance(sig, D8PSK)
    p = sig.dp
    rs, sps = p.rs, 8
    rate = rs * sps
    ev = _burst(sig, T0 + 0.01234, 230)
    k0 = sig.first_symbol(ev)
    assert k0 == math.ceil((T0 + 0.01234) * rs) and 0 <= k0 / rs - ev.start < 1 / rs  # on the absolute symbol grid
    n0 = (k0 - 10) * sps
    x = sig.render_events([ev], n0, (230 + 20) * sps, rate)
    s = sig.symbols(ev)
    got = x[10 * sps :: sps][:230]
    c = rc_taps(p.alpha, sps, p.span)[p.span * sps // 2]  # centre tap (unit-power scaling)
    np.testing.assert_allclose(got, c * s, atol=2e-4)  # raised cosine is Nyquist: symbol instants hold the symbols
    assert np.all(x[: (10 - p.span // 2) * sps] == 0) and np.all(x[(10 + 230 + p.span // 2) * sps + 1 :] == 0)
    steps = np.round(np.angle(s[1:] / s[:-1]) / (np.pi / 4)).astype(int) % 8
    frac = np.angle(s[1:] / s[:-1]) / (np.pi / 4)
    np.testing.assert_allclose(frac, np.round(frac), atol=1e-9)  # every step a multiple of 45 degrees
    np.testing.assert_array_equal(steps[p.ramp_up - 1 : p.ramp_up - 1 + len(SYNC)], GRAY[list(SYNC)])
    assert np.all(steps[: p.ramp_up - 1] == 0) and np.all(steps[-p.ramp_down :] == 0)
    amp = np.abs(s)
    assert np.all(np.diff(amp[: p.ramp_up]) > 0) and amp[0] < 0.1 and np.all(np.diff(amp[-p.ramp_down :]) < 0)
    np.testing.assert_allclose(amp[p.ramp_up : -p.ramp_down], 1.0)
    # any window is a slice of a longer one
    np.testing.assert_array_equal(sig.render_events([ev], n0 + 333, 500, rate), x[333:833])
    # keyed by the burst start
    twin = make_signal("d8psk", {}, _keys(), dsp)
    assert isinstance(twin, D8PSK)
    np.testing.assert_array_equal(twin.symbols(ev), s)
    assert not np.allclose(sig.symbols(_burst(sig, ev.start + 1, 230)), s)


def test_d8psk_power_and_bandwidth() -> None:
    sig = make_signal("d8psk", {}, _keys(), dsp)
    assert isinstance(sig, D8PSK)
    rs, sps = 10_500, 8
    rate = rs * sps
    ev = _burst(sig, T0, 4000)
    k0 = sig.first_symbol(ev)
    x = sig.render_events([ev], (k0 + 50) * sps, 3800 * sps, rate).astype(np.complex128)
    assert abs(np.mean(np.abs(x) ** 2) - 1) < 0.05  # unit mean power while on
    f = np.fft.fftfreq(len(x), 1 / rate)
    psd = np.abs(np.fft.fft(x)) ** 2
    out = np.abs(f) > rs * 1.6 / 2 + 300
    assert psd[out].sum() / psd.sum() < 1e-3  # raised cosine 0.6: within rs (1 + alpha) / 2
    assert sig.bandwidth_hz() == pytest.approx(16_800)
    assert sig.preferred_rate() == 42_000 and sig.rate_step() == 10_500


def test_d8psk_burst_lengths() -> None:
    p = D8pskParams.model_validate({"burst_ms": [20, 23]})
    g = np.random.default_rng(3)
    lo, hi = p.symbol_range()
    for _ in range(200):
        d, prm = D8PSK.draw_event(p, g)
        assert lo <= prm["symbols"] <= hi and d == pytest.approx(prm["symbols"] / 10_500)
        assert d <= D8PSK.max_duration_s(p)
    with pytest.raises(ValidationError, match="at least"):
        D8pskParams.model_validate({"burst_ms": 2.0})


# --- gfsk ---------------------------------------------------------------------------------------------------------


def test_gfsk_burst() -> None:
    """VDL Mode 4 style: constant envelope, +-2.4 kHz deviation, centre-peaked (M7: 136.925 MHz is ~2.7 kHz wide at
    -10 dB in the recording, VDL2 D8PSK ~11 kHz)."""
    sig = make_signal("gfsk", {}, _keys(), dsp)
    assert isinstance(sig, GFSK)
    p = sig.gp
    sps = 8
    rate = p.rb * sps
    ev = Event(T0 + 0.00321, T0 + 0.00321 + 20_000 / p.rb, {"bits": 20_000})
    k0 = sig.first_bit(ev)
    assert k0 == math.ceil((T0 + 0.00321) * p.rb)
    n0 = k0 * sps - 50
    x = sig.render_events([ev], n0, 20_000 * sps + 100, rate).astype(np.complex128)
    assert np.all(x[:50] == 0) and np.all(x[-50:] == 0)
    body = x[50 + 10 * sps : -50 - 10 * sps]
    np.testing.assert_allclose(np.abs(body), 1.0, atol=1e-5)  # constant envelope, unit power
    f_inst = measure.inst_freq(body, rate)
    dev = p.h * p.rb / 2
    assert np.max(np.abs(f_inst)) <= dev * 1.001 and np.percentile(np.abs(f_inst), 99) > 0.8 * dev
    assert 4500 < ter.mean_spectrum_width_db10(body, rate) < 6000  # random data
    np.testing.assert_array_equal(sig.render_events([ev], n0 + 4321, 999, rate), x[4321:5320].astype(np.complex64))
    assert sig.preferred_rate() == 76_800 and sig.rate_step() == 19_200
    # frequent transitions (NRZI-like) narrow it: the fit P14 makes to the recording's 136.925 MHz bursts
    narrow = make_signal("gfsk", {"transition_p": 0.665}, _keys(), dsp)
    y = narrow.render_events([ev], n0, 20_000 * sps + 100, rate).astype(np.complex128)[50 + 10 * sps : -50 - 10 * sps]
    assert abs(ter.mean_spectrum_width_db10(y, rate) - 2734) < 400


# --- am -----------------------------------------------------------------------------------------------------------


def test_am_signal() -> None:
    sig = make_signal("am", {}, _keys(), dsp)
    assert isinstance(sig, AM)
    rate = 9000
    ev = Event(T0 + 0.5, T0 + 4.5, {"duration_s": 4.0})
    n0 = round(T0 * rate)
    x = sig.render_events([ev], n0, 5 * rate, rate).astype(np.complex128)
    j0, j1 = round(0.5 * rate), round(4.5 * rate)
    assert np.all(x[: j0 - 1] == 0) and np.all(x[j1 + 2 :] == 0)
    assert abs(x[j0]) < 0.01 and abs(np.abs(x[j0 + round(0.03 * rate) : j1 - round(0.03 * rate)]).min()) > 0.15
    body = x[j0 + rate // 10 : j1 - rate // 10]
    assert abs(np.mean(body) - 1) < 0.03  # the carrier has unit amplitude (the level is the carrier's)
    f = np.fft.fftfreq(len(body), 1 / rate)
    psd = np.abs(np.fft.fft(body - np.mean(body))) ** 2 / len(body) ** 2
    assert psd[np.abs(f) > 3200].sum() / (1 + psd.sum()) < 1e-4  # voice within +-3 kHz (clipping: 1.6e-5)
    voice = psd[(np.abs(f) > 300) & (np.abs(f) < 3000)].sum()
    assert 0.01 < voice < 0.3  # sidebands a few per cent of the carrier
    # syllables and pauses
    env = sig.envelope(ev, np.arange(0, 4, 0.01))
    assert 0.1 < np.mean(env < 0.05) < 0.7
    np.testing.assert_array_equal(
        sig.render_events([ev], n0 + 12_345, 5000, rate), x[12_345:17_345].astype(np.complex64)
    )


def test_noise_band() -> None:
    sig = make_signal("noise_band", {"width_hz": 2200}, _keys(), dsp)
    rate = 6000
    n0 = round(T0 * rate)
    x = sig.render(n0, 30 * rate, rate).astype(np.complex128)
    assert abs(np.mean(np.abs(x) ** 2) - 1) < 0.05
    f, p = welch(x, fs=rate, nperseg=1200, return_onesided=False)
    half = f[p >= p.max() / 2]
    assert abs((half.max() - half.min()) - 2200) < 250
    np.testing.assert_array_equal(sig.render(n0 + 777, 1000, rate), x[777:1777].astype(np.complex64))


# --- schedules ----------------------------------------------------------------------------------------------------


def test_event_envelope_ramps() -> None:
    t = np.array([9.9975, 10.0, 10.0025, 15.0, 19.9975, 20.0, 20.0025])
    e = event_envelope(t, 10.0, 20.0, 0.005)
    np.testing.assert_allclose(e, [0, 0.5, 1, 1, 1, 0.5, 0], atol=1e-9)


def test_poisson_dead_time_and_slot_cache() -> None:
    sch = PoissonEvents(KeyedRng(4).scope("s", "e"), DurationDist("fixed", s=0.02), 7200, min_gap_s=0.05)
    ev = sch.events(T0, T0 + 5 * 3600)
    starts = np.array([e.start for e in ev])
    ends = np.array([e.end for e in ev])
    assert np.all(starts[1:] >= ends[:-1] + 0.05 - 1e-9)  # one transmitter: never overlapping
    kept = len(ev) / (5 * 3600)
    assert abs(kept - 2 / (1 + 2 * 0.07)) < 0.05  # dead time: rate / (1 + rate (duration + gap))
    assert len(sch._slots) <= SLOT_CACHE
    again = PoissonEvents(KeyedRng(4).scope("s", "e"), DurationDist("fixed", s=0.02), 7200, min_gap_s=0.05)
    assert again.events(T0 + 4 * 3600, T0 + 4 * 3600 + 60) == sch.events(T0 + 4 * 3600, T0 + 4 * 3600 + 60)


# --- engine -------------------------------------------------------------------------------------------------------


def test_per_event_offsets_and_levels(scene: Scene) -> None:
    em = _em(scene, "pe")
    base = em.rx_dbm
    assert base is not None
    evs = em.event_truth(T0, T0 + 600)
    assert len(evs) > 30
    for et in evs:
        off, lvl = et.event.params["offset_hz"], et.event.params["level_db"]
        assert -1000 <= off <= 1000 and off == round(off * 1000) / 1000 and -5 <= lvl <= 5
        assert et.f_rx_mhz == 137_300_000_000 + round(off * 1000)
        assert et.rx_dbm == pytest.approx(base + lvl)
    rd = Renderer(scene, noise=False)
    for et in [e for e in evs if e.event.start > T0 + 1][:3]:
        a, b = et.event.start + 0.05, et.event.end - 0.05
        y = rd.iq(sample_at(a, FS), round((b - a) * FS)).astype(np.complex128)
        f_rx = et.f_rx_mhz / 1000 - FC
        assert abs(measure.tone_freq(y, FS, f_rx, 150.0) - f_rx) < 0.05
        z, _, _ = measure.extract_band(y, FS, f_rx, 150.0)
        p = np.mean(np.abs(z[len(z) // 10 : -len(z) // 10]) ** 2)
        assert abs(10 * np.log10(p) - et.rx_dbm) < 0.05  # no profile: H_rx = 1


def test_ground_cn0_from_the_stream(scene: Scene) -> None:
    rd = Renderer(scene)
    rd.wide = []  # USB humps add to the noise density (by design); this checks the ground level alone
    t = T0 + 20
    iq = measure.iq_from_u8(rd.u8(sample_at(t, FS), 2 * FS))
    tr = _em(scene, "tone").truth(np.arange(t, t + 2, 0.5), 0.5)
    assert np.all(np.isnan(tr.el_deg)) and np.all(tr.doppler_mhz == 0)
    got, _, _ = measure.cn0_dbhz(iq, FS, 0.4e6 - 5, 0.4e6 + 5)
    assert abs(got - 45) < 0.3 and tr.cn0_dbhz[0] == pytest.approx(45)


def test_two_am_transmitters_beat(tmp_path: Path) -> None:
    one = {"rate_per_hour": 360, "duration": {"dist": "fixed", "s": 30}}
    srcs = [
        {"id": f"a{i}", "preset": "atc_am", "freq_hz": 136.3e6, "offset_hz": off, "schedule": one}
        | {"per_event": {"offset_hz": 0, "level_db": {"sigma_db": 0, "min_db": 0, "max_db": 0}}}
        for i, off in ((1, 0.0), (2, 700.0))
    ]
    sc = Scene(_resolve(tmp_path, srcs, "beat").scenario)
    e1, e2 = (_em(sc, s).schedule.events(T0, T0 + 3600) for s in ("a1", "a2"))
    t = next(max(a.start, b.start) + 0.5 for a in e1 for b in e2 if min(a.end, b.end) - max(a.start, b.start) > 2)
    y = Renderer(sc, noise=False).iq(sample_at(t, FS), FS).astype(np.complex128)
    z, rate, _ = measure.extract_band(y, FS, 136.3e6 - FC + 350, 4500.0)
    env = np.abs(z[len(z) // 10 : -len(z) // 10]) ** 2
    spec = np.abs(np.fft.rfft(env - env.mean()))
    f = np.fft.rfftfreq(len(env), 1 / rate)
    assert abs(f[np.argmax(spec[f > 50]) + np.count_nonzero(f <= 50)] - 700) < 3


def test_noise_comb(scene: Scene) -> None:
    em = _em(scene, "hum")
    eps = em.schedule.events(T0, T0 + 7200)
    ep = next(
        e for e in eps if e.start > T0 and not any(o is not e and o.end > e.start and o.start < e.end for o in eps)
    )
    t = ep.start + 10  # one episode alone
    c = math.floor(t)
    rd = Renderer(scene, noise=False)
    assert [w.id for w in rd.wide] == ["hum/comb"] and all(er.id != "hum/comb" for er in rd.emitters)
    rd.emitters = []  # the comb alone (a wideband emitter, added in the time domain)
    y = rd.signal(c)
    assert y is not None
    seg = y[: FS // 2].astype(np.complex128).reshape(-1, 2400)
    psd = np.mean(np.abs(np.fft.fft(seg, axis=1)) ** 2, axis=0) / (2400 * FS)  # mW/Hz
    d = round(FS / ep.params["spacing_hz"])
    n0_w = link.K_BOLTZ * em.t_sys_k * 1000
    beta = beta_of(ep.params["amplitude_db"])
    assert np.mean(psd) == pytest.approx(beta * n0_w, rel=0.05)  # beta k T on average
    f = np.fft.fftfreq(2400, 1 / FS)
    ph = 2 * np.pi * f * d / FS
    a = 2 * np.mean(psd * np.cos(ph)) / np.mean(psd)
    b = 2 * np.mean(psd * np.sin(ph)) / np.mean(psd)
    assert abs(math.hypot(a, b) - 1) < 0.1  # (1 + cos): a full-depth ripple with period fs / d


def test_window_independence_workers_and_keys(tmp_path: Path, resolved, scene: Scene) -> None:
    rd = Renderer(scene)
    n0 = N0 + 6 * FS + 123_457
    n = round(2.5 * FS)
    full = rd.u8(n0, n)
    np.testing.assert_array_equal(Renderer(scene).u8(n0 + 1_000_003, FS), full[2_000_006 : 2_000_006 + 2 * FS])
    parts = list(workers.iter_u8(resolved.canonical, ROOT, n0, n, FS, 2, task_s=1))
    np.testing.assert_array_equal(np.concatenate(parts), full)
    alone = Scene(_resolve(tmp_path, [s for s in SOURCES if s["id"] != "am"], "noam").scenario)
    a, b = Renderer(scene), Renderer(alone)
    c = int(T0) + 7
    for eid in ("gs/vdl", "air/vdl", "pe/beep"):
        sa = next(er for er in a.emitters if er.id == eid).stream(c)
        sb = next(er for er in b.emitters if er.id == eid).stream(c)
        assert (sa is None) == (sb is None)
        if sa is not None and sb is not None:
            np.testing.assert_array_equal(sa.x, sb.x)


# --- truth --------------------------------------------------------------------------------------------------------


def test_truth_of_terrestrial_sources(scene: Scene) -> None:
    table = T.compute(scene, T0, T0 + 20, 1.0)
    rows = [r for r in table.rows if r["source_id"] == "tone"]
    assert len(rows) == 20 and rows[0]["el_deg"] == "" and rows[0]["range_km"] == "" and rows[0]["norad"] == ""
    assert (
        rows[0]["doppler_hz"] == "0.000" and rows[0]["f_rx_hz"] == "137400000.000" and rows[0]["cn0_dbhz"] == "45.000"
    )
    vdl = [e for e in table.events if e["signal"] == "d8psk"]
    assert vdl and all(e["f_rx_hz"] and e["rx_dbm"] and e["cn0_dbhz"] for e in vdl)
    air = [e for e in vdl if e["source_id"] == "air"]
    assert air and len({e["f_rx_hz"] for e in air}) == len(air)  # every aircraft burst its own offset
    comb = [e for e in T.compute(scene, T0, T0 + 7200, 3600.0).events if e["signal"] == "noise_comb"]
    assert comb and comb[0]["f_rx_hz"] == "" and "amplitude_db" in comb[0]["params"]
    only_sat = T.compute(scene, T0, T0 + 20, 1.0, ("satellite",))
    assert not only_sat.rows and not only_sat.events


def test_truth_cli_platform_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    doc = {"seed": 1, "clock": {"start": "2026-11-04T12:00:00Z", "duration": "1min"}, "station": STATION}
    doc["sources"] = SOURCES[:1] + SOURCES[5:6]
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(doc))
    out = tmp_path / "truth.csv"
    r = CliRunner().invoke(app, ["truth", str(p), "-o", str(out), "--to", "+10s", "--platforms", "ground"])
    assert r.exit_code == 0, r.output
    assert out.read_text().count("\n") == 11 and (tmp_path / "truth.events.csv").is_file()
    bad = CliRunner().invoke(app, ["truth", str(p), "-o", str(out), "--platforms", "aircraft"])
    assert bad.exit_code != 0


# --- the band's terrestrial scenario ------------------------------------------------------------------------------


def test_band_file_is_derived_from_the_profile() -> None:
    text = (band_dir("vhf137") / "terrestrial_default.yaml").read_text()
    assert text == ter.band_yaml(ROOT / "profiles" / "vhf137")


def test_band_vdl_modulations() -> None:
    """M7: 136.925 MHz carries VDL Mode 4 GFSK (constant envelope, ~2.7 kHz wide), 136.975 MHz VDL Mode 2 D8PSK."""
    doc = yaml.safe_load((band_dir("vhf137") / "terrestrial_default.yaml").read_text())
    kinds = {(s["freq_hz"], s["signal"]["kind"]) for s in doc["sources"] if s["id"].startswith("vdl")}
    assert kinds == {(136.925e6, "gfsk"), (136.975e6, "d8psk")}


def test_band_terrestrial_scenario_renders(tmp_path: Path) -> None:
    doc = {
        "seed": 2,
        "clock": {"start": "2026-11-01T16:39:00Z", "duration": "1h"},
        "station": STATION,
        "profile": "profiles/vhf137",
        "include": ["bands/vhf137/terrestrial_default.yaml"],
    }
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(doc))
    sc = resolve_document(load_scenario(p), ROOT).scenario
    assert len(sc.sources) >= 20 and {s.platform for s in sc.sources} == {"ground", "local"}
    rd = Renderer(Scene(sc), ROOT)
    t = parse_iso("2026-11-01T16:50:00Z")
    u = rd.u8(sample_at(t, FS), FS).astype(np.int16)
    clip = np.mean(np.abs(u - 127.5) >= 126.5)
    assert 0 < clip < 0.02  # the strong VDL stations clip now and then
    assert rd.emitters and {er.id.split("/")[0] for er in rd.emitters} >= {"vdl-136.925-a", "carrier-138"}
