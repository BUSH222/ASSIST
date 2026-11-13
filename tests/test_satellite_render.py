"""The satellite signals through the whole engine: presets, levels, timing, the ORBCOMM estimators, window
independence and keyed draws.

One synthetic scene, all sources up at the start (no samples needed, flat noise shape: no profile):
- `met`: METEOR LRPT (meteor_lrpt preset) on 137.9 MHz;
- `oc`: ORBCOMM (orbcomm preset) on S-2 and S-8, gateway on;
- `sb`: LoRa (spacebee preset) on 137.099 MHz only, with short gaps so bursts fall in short windows.
Checks:
- clean signal power in each band = truth rx level (no profile: H_rx = 1), within 0.05 dB;
- C/N0 from the uint8 stream (noise and ADC) = truth within 0.3 dB;
- ORBCOMM subscriber through the profile tool's estimators: alpha 0.4 +- 0.02, 4800 sym/s within 0.01 %;
- gateway block edges and LoRa burst edges at the receiver within 0.1 ms of the schedule; LoRa chirp slope;
- window independence, worker processes and adding a source leave the bytes / other sources unchanged.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from scipy.signal import welch

from assist import measure
from assist.config.loader import ConfigError, load_scenario
from assist.config.resolve import resolve_document
from assist.engine import workers
from assist.engine.render import Renderer
from assist.engine.scene import Scene
from assist.profile import orbcomm as p2b
from assist.profile.dsp import channelize
from assist.timebase import parse_iso, sample_at

ROOT = Path(__file__).resolve().parent.parent
FS = 2_400_000
FC = 137.0e6
T0 = parse_iso("2026-11-04T12:00:00Z")
N0 = sample_at(T0, FS)
STATION = {"file": None, "lat": 55.5, "lon": 37.1, "alt_m": 172}

SOURCES: list[dict[str, Any]] = [
    {
        "id": "met",
        "preset": "meteor_lrpt",
        "orbit": {"synthetic_pass": {"aos": "-4min", "max_el_deg": 60, "direction": "desc", "side": "east"}},
        "level": {"cn0_zenith_dbhz": 62},  # moderate, so the ADC does not clip in the C/N0 check
    },
    {
        "id": "oc",
        "preset": "orbcomm",
        "orbit": {
            "synthetic_pass": {
                "aos": "-3min",
                "max_el_deg": 50,
                "direction": "asc",
                "side": "west",
                "alt_km": 750,
                "inc_deg": 70,
            }
        },
        "channels": ["S-2", "S-8"],
        "gateway": {"enabled": True},
    },
    {
        "id": "sb",
        "preset": "spacebee",
        "orbit": {
            "synthetic_pass": {
                "aos": "-5min",
                "max_el_deg": 70,
                "direction": "asc",
                "side": "east",
                "alt_km": 470,
                "inc_deg": 97.3,
            }
        },
        "lora": {"freq_hz": 137.099e6, "schedule": {"mean_gap_s": 12, "min_gap_s": 3}},
    },
]


def _resolve(tmp: Path, sources: list[dict[str, Any]], name: str = "sat"):
    doc = {"seed": 21, "clock": {"start": "2026-11-04T12:00:00Z", "duration": "10min"}, "station": STATION}
    doc["sources"] = sources
    p = tmp / f"{name}.yaml"
    p.write_text(yaml.safe_dump(doc))
    return resolve_document(load_scenario(p), ROOT)


@pytest.fixture(scope="module")
def resolved(tmp_path_factory: pytest.TempPathFactory):
    return _resolve(tmp_path_factory.mktemp("sat"), SOURCES)


@pytest.fixture(scope="module")
def scene(resolved) -> Scene:
    return Scene(resolved.scenario)


def _er(rd: Renderer, eid: str):
    return next(er for er in rd.emitters if er.id == eid)


def _truth_mean(scene: Scene, sid: str, eid: str, t0: float, t1: float) -> tuple[float, float, float]:
    """(mean rx level mW, mean f_rx Hz, mean C/N0 linear) over [t0, t1) from truth at 0.1 s steps."""
    em = next(e for e in scene.source(sid).emitters if e.id == eid)
    tr = em.truth(np.arange(t0, t1, 0.1), 0.1)
    return float(np.mean(10 ** (tr.rx_dbm / 10))), float(np.mean(tr.f_rx_hz)), float(np.mean(10 ** (tr.cn0_dbhz / 10)))


def _band_power(y: np.ndarray, f_lo: float, f_hi: float) -> float:
    """Power of y (mW for sqrt(mW) samples) in baseband [f_lo, f_hi] Hz."""
    x = np.fft.fft(y.astype(np.complex128))
    f = np.fft.fftfreq(len(y), 1 / FS)
    sel = (f >= f_lo) & (f <= f_hi)
    return float(np.sum(np.abs(x[sel]) ** 2) / len(y) ** 2)


def _lora_event(scene: Scene, min_s: float = 0.4):
    em = scene.source("sb").emitters[0]
    return next(e for e in em.schedule.events(T0 + 1, T0 + 300) if e.end - e.start >= min_s and e.start > T0 + 1)


# --- presets and resolve ---


def test_presets_resolve(resolved) -> None:
    sc = resolved.scenario
    met, oc, sb = sc.sources
    lrpt = met.emitters[0]
    assert lrpt.signal.kind == "oqpsk" and lrpt.freq_hz == 137.9e6 and lrpt.signal.params["framing"] == "meteor_cadu"
    assert oc.channels == [137.225e6, 137.7125e6]
    assert [e.freq_hz for e in oc.emitters] == [137.225e6, 137.7125e6, 137.56e6]
    assert all(e.channel is None for e in oc.emitters)  # resolved into freq_hz
    assert oc.emitters[2].enabled and oc.emitters[2].signal.kind == "oqpsk_burst"
    assert all("cal orbcomm" in str(e.lvl.resolved_from) for e in oc.emitters)
    assert sb.emitters[0].freq_hz == 137.099e6 and sb.emitters[0].signal.kind == "lora"
    # Resolving the resolved document changes nothing.
    again = resolve_document(sc.model_dump(mode="json"), ROOT)
    assert again.canonical == resolved.canonical


def test_orbcomm_channels_are_drawn_per_source(tmp_path: Path) -> None:
    orbit = {"synthetic_pass": {"aos": "+1min", "max_el_deg": 40, "inc_deg": 70, "alt_km": 750}}
    srcs = [{"id": f"oc{i}", "preset": "orbcomm", "orbit": orbit} for i in range(12)]
    a = _resolve(tmp_path, srcs, "a").scenario
    table = {137.2e6, 137.225e6, 137.25e6, 137.44e6, 137.46e6, 137.6625e6, 137.6875e6, 137.7125e6, 137.7375e6}
    table |= {137.8e6, 137.2875e6, 137.3125e6}
    for s in a.sources:
        assert s.channels is not None and len(set(s.channels)) == 2 and set(s.channels) <= table
        assert [e.freq_hz for e in s.emitters[:2]] == s.channels
    assert len({tuple(s.channels or []) for s in a.sources}) > 3
    gw = [s.emitters[2].enabled for s in a.sources]
    assert any(gw) and not all(gw)  # bernoulli(0.5)
    b = _resolve(tmp_path, list(reversed(srcs)), "b").scenario  # keyed by id, not by position
    assert {s.id: s.channels for s in a.sources} == {s.id: s.channels for s in b.sources}


def test_channel_errors(tmp_path: Path) -> None:
    orbit = {"synthetic_pass": {"aos": "+1min", "max_el_deg": 40, "inc_deg": 70, "alt_km": 750}}
    with pytest.raises(ConfigError, match="unknown channel 'S-13'"):
        _resolve(tmp_path, [{"id": "o", "preset": "orbcomm", "orbit": orbit, "channels": ["S-1", "S-13"]}])
    with pytest.raises(ValueError, match="channels has 1"):
        _resolve(tmp_path, [{"id": "o", "preset": "orbcomm", "orbit": orbit, "channels": ["S-1"]}])
    # An explicit freq_hz replaces the preset's channel.
    sc = _resolve(tmp_path, [{"id": "o", "preset": "orbcomm", "orbit": orbit, "stx1": {"freq_hz": 137.46e6}}])
    assert sc.scenario.sources[0].emitters[0].freq_hz == 137.46e6
    em = {"id": "e", "signal": {"kind": "cw"}, "freq_hz": [1e8, 2e8], "level": {"p_tx_dbw": 0}}
    bad = {"id": "c", "orbit": orbit, "emitters": [em]}
    with pytest.raises(ValueError, match="needs an event signal"):
        _resolve(tmp_path, [bad])


def test_meteor_auto_calibration(tmp_path: Path) -> None:
    src = {"id": "m3", "preset": "meteor_lrpt", "orbit": {"norad": 57166}}
    doc: dict[str, Any] = {"seed": 1, "clock": {"start": "2026-11-03T09:00:00Z", "duration": "1h"}}
    doc["station"] = STATION
    doc["elements"] = {"files": [str(ROOT / "tests/fixtures/omm_2026-11-02.csv")]}
    p = tmp_path / "m.yaml"
    for norad, cal in ((57166, "meteor_m2_3"), (41188, "meteor")):
        doc["sources"] = [{**src, "orbit": {"norad": norad}}]
        p.write_text(yaml.safe_dump(doc))
        lv = resolve_document(load_scenario(p), ROOT).scenario.sources[0].emitters[0].lvl
        assert str(lv.resolved_from).startswith(f"cal meteor_auto -> cal {cal} ")


# --- levels ---


def test_clean_signal_power_matches_truth(scene: Scene) -> None:
    """Noiseless signal chunk (sqrt(mW)): power in each signal's band = truth rx level (H_rx = 1 without a profile)."""
    rd = Renderer(scene, noise=False)
    c = int(T0) + 3
    y = rd.signal(c)
    assert y is not None
    for sid, eid, half in (("met", "lrpt", 60e3), ("oc", "stx1", 4e3), ("oc", "stx2", 4e3)):
        p_mw, f_rx, _ = _truth_mean(scene, sid, eid, c, c + 1)
        got = _band_power(y, f_rx - FC - half, f_rx - FC + half)
        assert abs(10 * np.log10(got / p_mw)) < 0.05, (sid, eid, 10 * np.log10(got / p_mw))
    ev = _lora_event(scene)
    c = int(ev.start) + 1 if ev.end - int(ev.start) > 1.5 else int(ev.start)
    y = rd.signal(c)
    assert y is not None
    a = max(0, round((ev.start - c) * FS)) + 100
    b = min(FS, round((ev.end - c) * FS)) - 100
    p_mw, f_rx, _ = _truth_mean(scene, "sb", "lora", c + a / FS, c + b / FS)
    got = _band_power(y[a:b], f_rx - FC - 70e3, f_rx - FC + 70e3)
    assert abs(10 * np.log10(got / p_mw)) < 0.05


def _cn0_measured(iq: np.ndarray, f_c: float, half: float, ring: tuple[float, float]) -> float:
    f, p = welch(iq, fs=FS, nperseg=1 << 14, return_onesided=False, scaling="density")
    df = f[1] - f[0]
    band = np.abs(f - f_c) <= half
    nz = (np.abs(f - f_c) >= half + ring[0]) & (np.abs(f - f_c) <= half + ring[1])
    n0 = float(np.mean(p[nz]))
    c = float(np.sum(p[band] - n0) * df)
    return float(10 * np.log10(c / n0))


def test_cn0_from_the_stream_matches_truth(scene: Scene) -> None:
    rd = Renderer(scene)
    t = T0 + 10
    iq = measure.iq_from_u8(rd.u8(sample_at(t, FS), 2 * FS))
    for sid, eid, half, ring in (("met", "lrpt", 60e3, (8e3, 30e3)), ("oc", "stx1", 4.5e3, (3e3, 12e3))):
        _, f_rx, cn0 = _truth_mean(scene, sid, eid, t, t + 2)
        got = _cn0_measured(iq, f_rx - FC, half, ring)
        assert abs(got - 10 * np.log10(cn0)) < 0.3, (sid, got, 10 * np.log10(cn0))
    ev = _lora_event(scene, 1.0)
    a, b = ev.start + 0.01, ev.end - 0.01
    iq = measure.iq_from_u8(rd.u8(sample_at(a, FS), round((b - a) * FS)))
    _, f_rx, cn0 = _truth_mean(scene, "sb", "lora", a, b)
    got = _cn0_measured(iq, f_rx - FC, 64e3, (8e3, 30e3))
    assert abs(got - 10 * np.log10(cn0)) < 0.3, ("lora", got, 10 * np.log10(cn0))


# --- ORBCOMM against the profile tool's reference ---


def test_orbcomm_through_the_p2b_estimators(scene: Scene) -> None:
    """The subscriber channel cut out at 50 kS/s like the profile tool's reference, Doppler removed with truth: alpha
    0.4 +- 0.02, 4800 sym/s +- 0.01 %."""
    rd = Renderer(scene)
    secs = 20
    n0 = sample_at(T0 + 20, FS)
    f_ch = 137.225e6

    def read(a: int, n: int) -> np.ndarray:
        return measure.iq_from_u8(rd.u8(n0 + a, n))

    x = channelize(read, 0, secs * FS, FS, f_ch - FC, p2b.DECIM)
    em = scene.source("oc").emitters[0]
    tt = np.arange(0, secs + 1, 0.5)
    f_track = em.truth(T0 + 20 + tt, 0.5).f_rx_hz - f_ch
    xc = p2b.derotate(x, tt, f_track)
    rs = p2b.symbol_rate(xc)
    pulse = p2b.fit_pulse(xc, rs)
    assert abs(rs - 4800) <= 0.48 and abs(pulse["alpha"] - 0.4) <= 0.02, (rs, pulse)


# --- timing at the receiver ---


def _envelope_edges(z: np.ndarray, rate: float, smooth: int) -> np.ndarray:
    """Times (s from z[0]) where the smoothed amplitude crosses half its on-level (half-amplitude points)."""
    sm = np.convolve(np.abs(z), np.ones(smooth) / smooth, mode="same")
    level = float(np.median(sm[sm > 0.5 * sm.max()]))
    d = np.diff((sm > 0.5 * level).astype(np.int8))
    i = np.flatnonzero(d)
    frac = (0.5 * level - sm[i]) / (sm[i + 1] - sm[i])  # linear interpolation between samples i and i + 1
    return (i + frac) / rate


def test_gateway_and_lora_timing_at_the_receiver(scene: Scene) -> None:
    rd = Renderer(scene, noise=False)
    t_a, secs = T0 + 30, 5
    n0 = sample_at(t_a, FS)

    def read(a: int, n: int) -> np.ndarray:
        y = rd.iq(n0 + a, n)
        return y * (1 / np.sqrt(np.mean(np.abs(y) ** 2)))

    z = channelize(read, 0, secs * FS, FS, 137.56e6 - FC, 20)  # 120 kS/s around the gateway
    edges = t_a + _envelope_edges(z, FS / 20, 24)  # 0.2 ms smoothing
    gw = _er(rd, "oc/gateway").signal
    lo, hi = t_a + 0.05, t_a + secs - 0.05
    want = np.array([x for ab in gw.on_intervals(lo, hi) for x in ab])
    want = want[(want > lo) & (want < hi)]
    got = edges[(edges > lo) & (edges < hi)]
    assert len(got) == len(want) > 40 and np.max(np.abs(got - want)) < 0.1e-3

    ev = _lora_event(scene)
    t_b = ev.start - 0.05
    n1 = sample_at(t_b, FS)
    n_lora = round((ev.end - ev.start + 0.1) * FS)

    def read2(a: int, n: int) -> np.ndarray:
        return rd.iq(n1 + a, n) * 1e6

    z = channelize(read2, 0, n_lora, FS, 137.099e6 - FC, 8)  # 300 kS/s
    e = t_b + _envelope_edges(z, FS / 8, 15)
    assert abs(e[0] - ev.start) < 0.1e-3 and abs(e[-1] - ev.end) < 0.1e-3
    sf = ev.params["sf"]
    t_sym = 2**sf / 125e3
    i0 = round((ev.start - t_b) * FS / 8)
    seg = z[i0 + 20 : i0 + round(t_sym * FS / 8) - 20].astype(np.complex128)
    fi = np.angle(seg[1:] * np.conj(seg[:-1])) * (FS / 8) / (2 * np.pi)
    slope = np.polyfit(np.arange(len(fi)) / (FS / 8), fi, 1)[0]
    assert abs(slope / (125e3 / t_sym) - 1) < 2e-3  # Doppler rate (~100 Hz/s) is negligible against bw / T


# --- determinism ---


def test_window_independence_and_workers(resolved, scene: Scene) -> None:
    rd = Renderer(scene)
    n0 = N0 + 4 * FS + 123_457
    n = round(2.5 * FS)
    full = rd.u8(n0, n)
    np.testing.assert_array_equal(Renderer(scene).u8(n0 + 1_000_003, FS), full[2_000_006 : 2_000_006 + 2 * FS])
    parts = list(workers.iter_u8(resolved.canonical, ROOT, n0, n, FS, 2, task_s=1))
    np.testing.assert_array_equal(np.concatenate(parts), full)


def test_adding_a_source_leaves_the_others_unchanged(tmp_path: Path, scene: Scene) -> None:
    alone = Scene(_resolve(tmp_path, [SOURCES[0], SOURCES[1]], "two").scenario)
    a, b = Renderer(scene), Renderer(alone)
    c = int(T0) + 5
    for eid in ("met/lrpt", "oc/stx1", "oc/gateway"):
        sa, sb = _er(a, eid).stream(c), _er(b, eid).stream(c)
        assert sa is not None and sb is not None
        np.testing.assert_array_equal(sa.x, sb.x)
