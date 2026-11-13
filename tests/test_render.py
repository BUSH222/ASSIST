"""Render engine: Doppler and level in the IQ, window independence, determinism, keyed draws, zombie keying, noise.

CI renders short windows spread across a pass (random access makes that equivalent to slicing a full render); a
full-pass check of a real render is in test_render_local.py. Scenarios use the rounded station position and the
committed profile (noise shape, sigma 26.7 counts).
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from assist import measure
from assist.config.loader import load_scenario
from assist.config.resolve import resolve_document
from assist.engine.control import RenderError
from assist.engine.render import Renderer
from assist.engine.scene import Scene
from assist.timebase import parse_iso, sample_at

ROOT = Path(__file__).resolve().parent.parent
FS = 2_400_000
START = "2026-11-04T12:00:00Z"
STATION = {"file": None, "lat": 55.5, "lon": 37.1, "alt_m": 172}
PASS = {"aos": "+1min", "max_el_deg": 45, "direction": "desc", "side": "east"}


def _scene(tmp: Path, sources: list[dict[str, Any]], name: str = "s", **extra: object) -> Scene:
    doc = {"seed": 7, "clock": {"start": START, "duration": "20min"}, "station": STATION, "sources": sources}
    doc.setdefault("profile", "profiles/vhf137")
    doc.update(extra)
    p = tmp / f"{name}.yaml"
    p.write_text(yaml.safe_dump(doc))
    return Scene(resolve_document(load_scenario(p), ROOT).scenario)


def _zombie(sid: str = "z1", freq: float = 137.35e6, **over: object) -> dict[str, Any]:
    return {"id": sid, "preset": "zombie", "orbit": {"synthetic_pass": PASS}, "freq_hz": freq, **over}


def _cw(sid: str, freq: float, p_tx: float | None = None, orbit: dict[str, Any] | None = None) -> dict[str, Any]:
    level = {"p_tx_dbw": p_tx} if p_tx is not None else {"cn0_zenith_dbhz": 50}
    return {
        "id": sid,
        "orbit": orbit or {"synthetic_pass": PASS},
        "emitters": [
            {
                "id": "carrier",
                "signal": {"kind": "cw"},
                "freq_hz": freq,
                "oscillator": {"pass_offset_sigma_hz": 50, "drift_hz_s": 0.2},
                "level": level,
            }
        ],
    }


@pytest.fixture(autouse=True)
def _cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)


@pytest.fixture(scope="module")
def tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("render")


@pytest.fixture(scope="module")
def zombie(tmp: Path) -> Scene:
    return _scene(tmp, [_zombie()], "zombie")


@pytest.fixture(scope="module")
def cw(tmp: Path) -> Scene:
    return _scene(tmp, [_cw("c1", 137.35e6)], "cw")


def _pass(scene: Scene, sid: str):
    (p,) = scene.source(sid).platform.passes(scene.t_start, scene.t_end)
    return p


def _truth_freq(scene: Scene, sid: str, t: np.ndarray) -> np.ndarray:
    tr = scene.source(sid).emitters[0].truth(t, 1e-3)
    assert tr.t.size == t.size
    return tr.f_rx_hz - scene.scenario.frontend.fc_hz


def _freq_errors(scene: Scene, sid: str, starts: list[float], dur: float = 0.2) -> np.ndarray:
    """Rendered instantaneous frequency (noiseless, 1 ms means) minus truth f_rx, over windows at `starts`."""
    rd = Renderer(scene, noise=False)
    errs = []
    seg = FS // 1000
    for t0 in starts:
        n0 = sample_at(t0, FS)
        y = rd.counts(n0, round(dur * FS))
        f = measure.inst_freq(y, FS)
        k = len(f) // seg
        fm = f[: k * seg].reshape(k, seg).mean(axis=1)
        tm = (n0 + np.arange(k) * seg + seg / 2) / FS
        errs.append(fm - _truth_freq(scene, sid, tm))
    return np.concatenate(errs)


def test_doppler_from_iq_over_a_pass(cw: Scene) -> None:
    """Doppler in the IQ: instantaneous frequency of the rendered CW vs truth f_rx, <= 0.1 Hz RMS across the pass."""
    p = _pass(cw, "c1")
    starts = [*np.linspace(p.aos + 0.5, p.los - 1.0, 12), p.tca - 0.1]
    err = _freq_errors(cw, "c1", starts)
    assert np.sqrt(np.mean(err**2)) <= 0.1
    assert np.max(np.abs(err)) < 0.01  # in fact ~1 mHz: f_rx truth omits the (df_tx/dt) R / c term


def test_zombie_carrier_from_iq(zombie: Scene) -> None:
    """The zombie's per-pass offset and drift (and AM, keying) render as truth says, in its on intervals."""
    p = _pass(zombie, "z1")
    sch = zombie.source("z1").emitters[0].schedule
    on = [(a, b) for a, b in sch.on_intervals(p.aos, p.los) if a > p.aos and b < p.los]  # ty: ignore[unresolved-attribute]
    assert len(on) >= 3
    starts = [a + 0.5 for a, _ in on] + [b - 0.7 for _, b in on]
    err = _freq_errors(zombie, "z1", starts)
    assert np.sqrt(np.mean(err**2)) <= 0.1
    em = zombie.source("z1").emitters[0]
    off, drift = em.carrier.pass_terms(p.aos_key)
    assert off != 0 and abs(drift) == 0.2


LEVEL_CARRIERS = (("lo", 135.9e6), ("mid", 137.1e6), ("hi", 137.9e6))  # SAW notch, passband, METEOR frequency


@pytest.fixture(scope="module")
def levels(tmp: Path) -> Scene:
    return _scene(tmp, [_cw(sid, f, p_tx=-12.0) for sid, f in LEVEL_CARRIERS], "levels")


def _truth_window(scene: Scene, sid: str, n0: int, n: int):
    tm = (n0 + np.arange(0, n, 2400) + 1200) / FS  # 1 ms steps
    return scene.source(sid).emitters[0].truth(tm, 1e-3)


def test_signal_level_chain_is_exact(levels: Scene) -> None:
    """Clean carrier power in counts = truth rx power * |H_rx(f)|^2 * G^2, to 0.01 dB, at low and high elevation."""
    from assist.engine.frontend import design_fir, fir_response

    rd = Renderer(levels, noise=False)
    h = fir_response(design_fir(rd.noise_spec.shape_db), FS)  # |H_rx| on a 1 Hz grid
    p = _pass(levels, "lo")
    for t0 in (p.aos + 20.0, p.tca - 0.5):
        n0 = sample_at(t0, FS)
        y = rd.counts(n0, FS)
        for sid, _f in LEVEL_CARRIERS:
            tr = _truth_window(levels, sid, n0, FS)
            f = tr.f_rx_hz - 137e6
            _cn0, c, _n0 = measure.cn0_dbhz(y, FS, float(f.min()) - 20, float(f.max()) + 20)
            want = rd.gain**2 * np.mean(10 ** (tr.rx_dbm / 10)) * h[round(float(f.mean())) % FS] ** 2
            assert 10 * np.log10(c / want) == pytest.approx(0.0, abs=0.01)


def test_levels_from_iq(levels: Scene) -> None:
    """Level: C/N0 measured from the uint8 stream (noise and ADC on) vs truth, within 0.3 dB.

    Carriers at three points of the noise shape, early in the pass (~41 dB-Hz) and at culmination (~57 dB-Hz);
    2 s windows keep the measurement's own scatter near 0.1 dB. The receiver response applies to signals and noise
    alike, so the calibrated C/N0 holds wherever the carrier is.
    """
    rd = Renderer(levels)
    p = _pass(levels, "lo")
    diffs = []
    for t0 in (p.aos + 60.0, p.tca - 1.0):
        n0 = sample_at(t0, FS)
        y = measure.iq_from_u8(rd.u8(n0, 2 * FS))
        for sid, _f in LEVEL_CARRIERS:
            tr = _truth_window(levels, sid, n0, 2 * FS)
            f = tr.f_rx_hz - 137e6
            got, _c, _n0 = measure.cn0_dbhz(y, FS, float(f.min()) - 20, float(f.max()) + 20, ring=(300, 3000))
            want = 10 * np.log10(np.mean(10 ** (tr.cn0_dbhz / 10)))
            assert 40 < want < 60
            diffs.append(got - want)
    assert np.max(np.abs(diffs)) <= 0.3, diffs


def test_fades_after_aos_and_before_los(cw: Scene, tmp: Path) -> None:
    """A satellite carrier rises from nothing to its link level over 15 s after AOS and falls back over 15 s before
    LOS (raised-cosine amplitude), the same in every window; a pass shorter than 2 fade_s never reaches full level;
    fade_s = 0 gives the full level at once."""
    p = _pass(cw, "c1")
    em = cw.source("c1").emitters[0]
    assert em.fade_s == 15.0
    rd = Renderer(cw, noise=False)

    def gain2(t0: float, scene: Scene = cw, rd: Renderer = rd) -> float:
        n0 = sample_at(t0, FS)
        y = rd.counts(n0, FS // 20)
        t = (n0 + FS // 40) / FS
        rx = float(scene.source("c1").emitters[0].truth(np.array([t]), 1e-3).rx_dbm[0])
        return float(np.mean(np.abs(y) ** 2)) / math.pow(10.0, rx / 10)

    def want(x: float) -> float:
        return (0.5 - 0.5 * math.cos(math.pi * x)) ** 2

    ref = gain2(p.aos + 40.0)
    for dt in (3.0, 7.5, 12.0):
        assert gain2(p.aos + dt) / ref == pytest.approx(want(dt / 15.0), rel=0.05)
        assert gain2(p.los - 0.05 - dt) / ref == pytest.approx(want(dt / 15.0), rel=0.05)
    assert gain2(p.aos + 15.5) / ref == pytest.approx(1.0, rel=0.02)
    assert gain2(p.los - 15.6) / ref == pytest.approx(1.0, rel=0.02)
    assert gain2(p.aos + 0.2) / ref < 1e-3
    assert gain2(p.los - 0.3) / ref < 1e-3
    again = Renderer(Scene(cw.scenario), noise=False)
    assert gain2(p.aos + 7.5, cw, again) == gain2(p.aos + 7.5)
    assert gain2(p.los - 7.5, cw, again) == gain2(p.los - 7.5)

    # Fade longer than half the pass: the carrier peaks below full level, at x = (pass length / 2) / fade_s.
    d = p.los - p.aos
    long = _scene(tmp, [_cw("c1", 137.35e6)], "longfade", station={**STATION, "fade_s": d})
    mid = p.aos + d / 2
    assert gain2(mid, long, Renderer(long, noise=False)) / gain2(mid) == pytest.approx(want(0.5), rel=0.05)

    off = _scene(tmp, [_cw("c1", 137.35e6)], "nofade", station={**STATION, "fade_s": 0})
    assert off.source("c1").emitters[0].fade_s == 0.0
    roff = Renderer(off, noise=False)
    assert gain2(p.aos + 0.5, off, roff) / ref == pytest.approx(1.0, rel=0.05)
    assert gain2(p.los - 0.6, off, roff) / ref == pytest.approx(1.0, rel=0.05)


def test_render_window_independence(zombie: Scene, tmp: Path) -> None:
    """Window independence and determinism: a window rendered on its own equals the slice of a longer render, byte for
    byte.

    The windows cross a 1 s control chunk and an on/off edge, and the inner window starts off the hop grid.
    """
    sch = zombie.source("z1").emitters[0].schedule
    p = _pass(zombie, "z1")
    t_on = next(a for a, _ in sch.on_intervals(p.aos, p.los) if a > p.aos + 2)  # ty: ignore[unresolved-attribute]
    n_a = sample_at(t_on - 1.3, FS)
    n_b = sample_at(math.ceil(t_on - 1.3) + 0.3333337, FS)
    n_end = sample_at(t_on + 0.9, FS)
    full = Renderer(Scene(zombie.scenario)).u8(n_a, n_end - n_a)
    part = Renderer(Scene(zombie.scenario)).u8(n_b, n_end - n_b)
    assert n_b % (FS // 100) != 0
    assert np.array_equal(part, full[2 * (n_b - n_a) :])
    again = Renderer(Scene(zombie.scenario)).u8(n_a, n_end - n_a)
    assert np.array_equal(again, full)


def test_adding_a_source_changes_no_other_signal(tmp: Path) -> None:
    """Keyed draws: another source changes neither the first source's narrow samples nor, while it is silent, the
    output bytes."""
    a = _scene(tmp, [_zombie()], "key_a")
    later = dict(PASS, aos="+40min")
    ab = _scene(tmp, [_zombie(), _cw("c2", 136.5e6), _zombie("z9", 137.6e6, orbit={"synthetic_pass": later})], "ab")
    p = _pass(a, "z1")
    n0 = sample_at(p.tca, FS)
    ra, rab = Renderer(a), Renderer(ab)
    for c in range(n0 // FS - 30, n0 // FS + 30, 3):
        va, vab = ra.emitters[0].stream(c), rab.emitters[0].stream(c)
        assert (va is None) == (vab is None)
        if va is not None and vab is not None:
            assert np.array_equal(va.x, vab.x) and np.array_equal(va.active, vab.active)
    a2 = _scene(tmp, [_zombie()], "key_a2")
    b2 = _scene(tmp, [_zombie(), _zombie("z9", 137.6e6, orbit={"synthetic_pass": later})], "key_b2")
    np.testing.assert_array_equal(Renderer(a2).u8(n0, 50_000), Renderer(b2).u8(n0, 50_000))


def test_on_off_edges_in_iq(zombie: Scene) -> None:
    """Zombie keying: the half-amplitude points of the rendered envelope are the schedule's edges (+-20 us)."""
    sch = zombie.source("z1").emitters[0].schedule
    p = _pass(zombie, "z1")
    rd = Renderer(zombie, noise=False)
    edges = [(a, b) for a, b in sch.on_intervals(p.aos, p.los) if a > p.aos and b < p.los][:2]  # ty: ignore[unresolved-attribute]
    for t_on, t_off in edges:
        for t_edge, rising in ((t_on, True), (t_off, False)):
            n0 = sample_at(t_edge - 0.01, FS)
            env = np.abs(rd.counts(n0, FS // 50))  # 20 ms around the edge
            level = env[-1000:].mean() if rising else env[:1000].mean()
            above = env > level / 2
            i = int(np.argmax(above if rising else ~above))
            t_half = (n0 + i) / FS
            assert abs(t_half - t_edge) < 20e-6
            quiet = env[:100] if rising else env[-100:]
            assert quiet.max() < 1e-3 * level  # off outside the 5 ms ramp


def test_noise_level_and_shape(tmp: Path) -> None:
    """Noise only: sigma per component is the profile's 26.7 counts; the PSD follows the receiver response."""
    scene = _scene(tmp, [], "quiet")
    rd = Renderer(scene)
    n0 = sample_at(scene.t_start + 1.5, FS)
    x = measure.iq_from_u8(rd.u8(n0, 2 * FS))
    sigma = rd.noise_spec.sigma_counts
    assert np.std(x.real) == pytest.approx(sigma, rel=0.01)
    assert np.std(x.imag) == pytest.approx(sigma, rel=0.01)
    assert abs(np.mean(x.real)) < 0.05 and abs(np.mean(x.imag)) < 0.05
    # PSD in 16384 bins, averaged over 64-bin groups (9.4 kHz), vs the receiver response on the same grid.
    nfft = 16384
    seg = x[: (len(x) // nfft) * nfft].reshape(-1, nfft) * np.hanning(nfft)
    psd = np.fft.fftshift(np.mean(np.abs(np.fft.fft(seg, axis=1)) ** 2, axis=0))
    from assist.engine.frontend import design_fir, fir_response

    resp = np.fft.fftshift(fir_response(design_fir(rd.noise_spec.shape_db), nfft) ** 2)
    g = lambda a: a.reshape(-1, 64).mean(axis=1)  # noqa: E731
    diff = 10 * np.log10(g(psd) / g(resp))
    diff -= np.median(diff)
    assert np.sqrt(np.mean(diff**2)) < 0.1


def test_flat_noise_without_profile(tmp: Path) -> None:
    scene = _scene(tmp, [], "flat", profile=None)
    rd = Renderer(scene)
    assert rd.front.h_rx is None and rd.noise_spec.source.startswith("band")
    x = measure.iq_from_u8(rd.u8(sample_at(scene.t_start, FS), FS // 2))
    assert np.std(x.real) == pytest.approx(26.687, rel=0.02)


def test_excursion_beyond_the_narrow_rate_fails_loudly(cw: Scene) -> None:
    rd = Renderer(Scene(cw.scenario), noise=False)
    er = rd.emitters[0]
    er.limit_hz = 100.0  # pretend the rate were too low for the Doppler
    p = _pass(cw, "c1")
    with pytest.raises(RenderError, match="excursion"):
        rd.counts(sample_at(p.aos + 30.0, FS), 1000)  # Doppler ~ +3 kHz here (near 0 at culmination)


def test_narrow_rate_and_bin(cw: Scene) -> None:
    er = Renderer(cw).emitters[0]
    assert er.rate % 100 == 0 and er.n == er.rate // 50 and er.n % 2 == 0
    assert er.f_bin_mhz == 137_350_000_000  # 137.35 MHz is on the 50 Hz grid
    assert 1.25 * 2 * 4000 <= er.rate <= 12_000  # Doppler (~3.6 kHz) + 6 sigma + drift, with margin


def test_ppm_must_be_zero(tmp: Path) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="only 0 is supported"):
        _scene(tmp, [], "ppm", frontend={"ppm": 1.5})


def test_unknown_signal_kinds_are_refused(tmp: Path) -> None:
    src = _cw("o1", 137.9e6)
    src["emitters"][0]["signal"] = {"kind": "bank_clip"}
    with pytest.raises(ValueError, match="unknown signal kind 'bank_clip'"):
        _scene(tmp, [src], "bank_clip")


def test_cpp_and_numpy_engines_agree(zombie: Scene) -> None:
    """The same render through the C++ kernels and their numpy twins: the noise draw is the same bits, the rest
    agrees to float rounding, so bytes differ by at most one count, and rarely."""
    from assist import _dsp, dsp_ref

    sch = zombie.source("z1").emitters[0].schedule
    p = _pass(zombie, "z1")
    t_on = next(a for a, _ in sch.on_intervals(p.aos, p.los) if a > p.aos + 2)  # ty: ignore[unresolved-attribute]
    n0 = sample_at(t_on - 0.4, FS)  # across the rising edge and a chunk boundary
    a = Renderer(Scene(zombie.scenario), dsp=_dsp).u8(n0, FS).astype(np.int16)
    b = Renderer(Scene(zombie.scenario), dsp=dsp_ref).u8(n0, FS).astype(np.int16)
    d = np.abs(a - b)
    assert d.max() <= 1
    assert np.count_nonzero(d) <= 1e-4 * d.size


def test_stage_times_are_recorded(cw: Scene) -> None:
    rd = Renderer(Scene(cw.scenario))
    rd.u8(sample_at(_pass(cw, "c1").tca, FS), FS // 2)
    for stage in ("control", "synth", "noise", "adc"):
        assert rd.times[stage] > 0
    assert rd.times["prefetch"] == 0


def test_prefetch_computes_the_next_hour_and_changes_nothing(tmp: Path) -> None:
    """Live serving: in the last minute of an hour the next hour's passes are computed a few emitters per chunk;
    the stream is the same bytes as without prefetching."""
    sources = [_cw(f"c{i}", 136.2e6 + i * 0.3e6, orbit={"synthetic_pass": dict(PASS, aos="+50min")}) for i in range(3)]
    scene = _scene(tmp, sources, "prefetch", clock={"start": "2026-11-04T12:00:00Z", "duration": "2h"})
    n0 = sample_at(parse_iso("2026-11-04T12:59:50Z"), FS)
    pre = Renderer(Scene(scene.scenario), prefetch=True)
    lazy = Renderer(Scene(scene.scenario))
    a = pre.u8(n0, 5 * FS)  # 12:59:50 .. 12:59:55, inside the last minute
    assert all(er.has_hour(math.floor(parse_iso("2026-11-04T13:00:00Z") / 3600)) for er in pre.emitters)
    assert pre.times["prefetch"] > 0
    np.testing.assert_array_equal(a, lazy.u8(n0, 5 * FS))
    np.testing.assert_array_equal(pre.u8(n0 + 5 * FS, 10 * FS), lazy.u8(n0 + 5 * FS, 10 * FS))  # across 13:00
