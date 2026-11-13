"""Satellite signals against real orbits and the recording's ORBCOMM reference (local only: they need the element
snapshots in samples/).

1. `scenarios/meteor_lora.yaml`: METEOR-M2 3 and M2-4 on 137.9 MHz, back to back, with Starlink-borne LoRa on the
   overlapping channel (137.9265 MHz). Rendered by random access in short windows across both passes:
   - METEOR C/N0 from the uint8 stream (outside LoRa bursts) = truth within 0.3 dB, where truth >= 40 dB-Hz;
   - METEOR's 4th-power line (ADRIFT's suppressed-carrier branch) at 4 (f_rx - f) of truth, within 5 Hz in carrier
     units (0.5 s windows: the Doppler rate smears it), > 15 dB over the spectrum's median;
   - the LoRa bursts inside the METEOR passes (7) are in the stream under METEOR: dechirping a burst's preamble
     gives a peak at its received frequency (truth, within 40 Hz) > 6 dB over the median, for every burst whose
     symbol energy over (N0 + METEOR's spectral density) is >= 4 dB; weaker ones (from Starlinks near the horizon)
     may be lost under METEOR, as they would be on air.
2. ORBCOMM FM117 in the recording's reference window (2026-11-01 17:02:25 UTC + 169 s, 0-10.6 deg):
   the orbcomm preset on S-5 and S-8 with FM117's measured transmit offsets, through the same analysis as the
   recording: symbol rate 4800 +- 0.48, alpha 0.4 +- 0.02, Doppler residual RMS <= 1.5 Hz (the recording: 0.9), rest
   frequency within 1 Hz of truth; median C/N0 within 2 dB of the recording's (which fades by 3 dB RMS about the model,
   the simulation does not). Level chain: the estimator's bias b against the simulation's truth is < 1 dB, and the
   level fit on the simulated blocks returns the `orbcomm` link constant that went in (191.6 dB-Hz) plus b, within 0.05
   dB.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from assist import _dsp, measure
from assist.config.loader import load_scenario
from assist.config.resolve import resolve_document
from assist.emitters import link
from assist.engine.render import Renderer
from assist.engine.scene import Scene
from assist.orbits.elements import load_newest
from assist.profile import orbcomm as p2b
from assist.profile.dsp import channelize, rc_psd
from assist.station import load_station
from assist.timebase import parse_iso, sample_at

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
ELEMENTS = [SAMPLES / "active.csv", SAMPLES / "satnogs.csv"]
pytestmark = pytest.mark.skipif(
    not all(p.is_file() for p in ELEMENTS) or not (SAMPLES / "station.yaml").is_file(),
    reason="needs samples/active.csv, samples/satnogs.csv and samples/station.yaml",
)

FS = 2_400_000
FC = 137.0e6
METEOR_HZ = 137.9e6
LORA_HZ = 137.9265e6


@pytest.fixture(scope="module", autouse=True)
def _cwd() -> None:
    import os

    os.chdir(ROOT)


@pytest.fixture(scope="module")
def meteor_scene() -> Scene:
    return Scene(resolve_document(load_scenario(ROOT / "scenarios/meteor_lora.yaml"), ROOT).scenario)


def _cn0(iq: np.ndarray, f_c: float, half: float, ring: tuple[float, float], rd: Renderer) -> float:
    """C/N0 (dB-Hz) of a signal in f_c +- half (baseband), N0 from rings beside it. The PSD is first divided by the
    receiver response |H_rx|^2 (the profile's noise shape), which shapes signals and noise alike, so a sloped shape
    does not bias N0."""
    from scipy.signal import welch

    f, p = welch(iq, fs=FS, nperseg=1 << 14, return_onesided=False, scaling="density")
    h = rd.front.h_rx
    if h is not None:
        fh = np.fft.fftshift(np.fft.fftfreq(len(h), 1 / FS))
        p = p / np.interp(f, fh, np.fft.fftshift(h.astype(np.float64)) ** 2)
    band = np.abs(f - f_c) <= half
    nz = (np.abs(f - f_c) >= half + ring[0]) & (np.abs(f - f_c) <= half + ring[1])
    n0 = float(np.mean(p[nz]))
    return float(10 * np.log10(np.sum(p[band] - n0) * (f[1] - f[0]) / n0))


def _channel(rd: Renderer, n0: int, n: int, f_hz: float, decim: int) -> np.ndarray:
    def read(a: int, k: int) -> np.ndarray:
        return measure.iq_from_u8(rd.u8(n0 + a, k))

    return channelize(read, 0, n, FS, f_hz - FC, decim)


def test_meteor_pair_with_lora_on_the_overlapping_channel(meteor_scene: Scene) -> None:
    sc = meteor_scene
    rd = Renderer(sc)
    lora = [
        (src.id, e)
        for src in sc.sources
        if src.id.startswith("sl-")
        for em in src.emitters
        for e in em.event_truth(sc.t_start, sc.t_end)
    ]
    busy = [(e.event.start - 0.1, e.event.end + 0.1) for _, e in lora]
    n_cn0 = n_line = 0
    bursts: list[tuple[str, float, float, float]] = []
    for sid in ("m2-3", "m2-4"):
        em = sc.source(sid).emitters[0]
        (p,) = sc.source(sid).platform.passes(sc.t_start, sc.t_end)
        for t in np.arange(p.aos + 30, p.los - 30, 45.0):
            tr = em.truth(np.array([t, t + 0.5, t + 1, t + 1.5]), 0.5)
            cn0 = float(10 * np.log10(np.mean(10 ** (tr.cn0_dbhz / 10))))
            f_rx = float(np.mean(tr.f_rx_hz))
            if cn0 >= 40 and not any(a < t + 2 and b > t for a, b in busy):
                iq = measure.iq_from_u8(rd.u8(sample_at(t, FS), 2 * FS))
                got = _cn0(iq, f_rx - FC, 60e3, (8e3, 30e3), rd)
                assert abs(got - cn0) < 0.3, (sid, t, got, cn0)
                n_cn0 += 1
            # 4th-power line on 0.5 s, 300 kS/s around 137.9 MHz
            z = _channel(rd, sample_at(t, FS), FS // 2, METEOR_HZ, 8).astype(np.complex128)
            z4 = (z / np.abs(z).mean()) ** 4
            spec = np.abs(np.fft.fft(z4 * np.hanning(len(z4)), 1 << 19)) ** 2
            fr = np.fft.fftfreq(len(spec), 8 / FS)
            k = int(np.argmax(spec))
            f_line = fr[k] / 4 + METEOR_HZ
            f_mid = float(em.truth(np.array([t + 0.25]), 1.0).f_rx_hz[0])
            if 10 * np.log10(spec[k] / np.median(spec)) > 15:
                assert abs(f_line - f_mid) < 5, (sid, t, f_line - f_mid)
                n_line += 1
        # LoRa bursts inside this METEOR pass, found under METEOR by dechirping their preamble
        for src_id, e in lora:
            ev = e.event
            if not (p.aos < ev.start and ev.end < p.los):
                continue
            mid = np.array([(ev.start + ev.end) / 2])
            cn0_met = float(em.truth(mid, 1.0).cn0_dbhz[0])
            es_n0 = _lora_es_n0(e.rx_dbm, int(ev.params["sf"]), cn0_met, sc.t_sys_k)
            peak_db, df = _dechirp(rd, ev.start, int(ev.params["sf"]), e.f_rx_mhz / 1000 - LORA_HZ)
            bursts.append((src_id, es_n0, peak_db, df))
    assert n_cn0 >= 15 and n_line >= 15
    assert len(bursts) == 7  # 2 bursts in the M2-3 pass, 5 in the M2-4 pass (seed 6)
    # Found at the received frequency of truth when strong enough under METEOR; noise alone tops out ~4.3 dB over
    # the median (the largest of ~2e4 bins, each a sum of 7 exponentials).
    for src_id, es_n0, peak_db, df in bursts:
        if es_n0 >= 4:
            assert peak_db > 6 and abs(df) < 40, (src_id, es_n0, peak_db, df)
    assert sum(es >= 4 for _, es, _, _ in bursts) >= 3


def _lora_es_n0(rx_dbm: float, sf: int, cn0_meteor: float, t_sys_k: float) -> float:
    """Symbol energy over (N0 + METEOR's spectral density) in dB. The chirp sweeps 125 kHz, so dechirping spreads
    METEOR's power: its RRC spectrum (height C / rs in the flat part) is averaged over the LoRa band, 26.5 kHz above
    METEOR's centre (Doppler moves both alike)."""
    cn0 = rx_dbm - 30 - link.kt_dbw_hz(t_sys_k)
    f = np.linspace(-62.5e3, 62.5e3, 2001) + (LORA_HZ - METEOR_HZ)
    rel = 10 ** ((cn0_meteor - 10 * np.log10(72e3)) / 10) * float(np.mean(rc_psd(f, 72e3, 0.6)))
    return float(cn0 + 10 * np.log10(2**sf / 125e3) - 10 * np.log10(1 + rel))


def _dechirp(rd: Renderer, start: float, sf: int, f_off: float) -> tuple[float, float]:
    """(peak over median in dB, peak frequency minus f_off in Hz) of the burst's first 7 preamble chirps, dechirped
    against references aligned to each symbol's exact start and summed noncoherently."""
    rate = FS / 8
    t_sym = 2**sf / 125e3
    n0 = sample_at(start, FS)
    z = _channel(rd, n0, round(8 * t_sym * FS) + 64, LORA_HZ, 8)
    t_z0 = n0 / FS - start  # z[0] relative to the burst start
    n = int(t_sym * rate) - 2
    acc = np.zeros(4 * n)
    for s in range(7):
        j0 = int(np.ceil((s * t_sym - t_z0) * rate))
        ref = _dsp.chirp(
            np.array([0.0]),
            np.array([t_sym]),
            np.array([0], np.int32),
            sf,
            125e3,
            j0 / rate + t_z0 - s * t_sym,
            1 / rate,
            n,
        )
        acc += np.abs(np.fft.fft(z[j0 : j0 + n] * np.conj(ref), 4 * n)) ** 2
    fr = np.fft.fftfreq(len(acc), 1 / rate)
    k = int(np.argmax(acc))
    return float(10 * np.log10(acc[k] / np.median(acc))), float(fr[k] - f_off)


def test_orbcomm_fm117_through_the_m1_p2b_analysis(tmp_path: Path) -> None:
    ref = __import__("json").loads((ROOT / "profiles/vhf137/orbcomm_ref.json").read_text())
    rest = [c["doppler"]["rest_freq_hz"] for c in ref["channels"]]
    t0 = parse_iso("2026-11-01T16:39:00Z") + ref["segment_s"][0]  # the recording's start (SDR++ name, UTC+3)
    dur = ref["segment_s"][1] - ref["segment_s"][0]
    doc = {
        "seed": 3,
        "clock": {"start": "2026-11-01T17:02:25Z", "duration": f"{dur + 2}s"},
        "profile": "profiles/vhf137",
        "station": {"fade_s": 0},  # the level chain, not the fades: the recorded segment starts at AOS
        "elements": {"files": [str(p) for p in ELEMENTS]},
        "sources": [
            {
                "id": "fm117",
                "preset": "orbcomm",
                "orbit": {"norad": 41188},
                "channels": ["S-5", "S-8"],
                "stx1": {"offset_hz": round(rest[0] - 137.46e6, 3)},
                "stx2": {"offset_hz": round(rest[1] - 137.7125e6, 3)},
                "gateway": {"enabled": False},  # the recording: no gateway in the segment (+0.44 dB at 137.56 MHz)
            }
        ],
    }
    p = tmp_path / "fm117.yaml"
    p.write_text(yaml.safe_dump(doc))
    sc = Scene(resolve_document(load_scenario(p), ROOT).scenario)
    rd = Renderer(sc)
    n0 = sample_at(t0, FS)
    n = int(dur * FS)
    row = load_newest(ELEMENTS)[41188]
    st = load_station(SAMPLES / "station.yaml")
    sims = []
    for i, c in enumerate(ref["channels"]):
        x = _channel(rd, n0, n, c["channel_centre_hz"], p2b.DECIM)
        r = p2b.analyse(x, t0, c["channel_centre_hz"], row, st)
        assert abs(r["symbol_rate_hz"] - 4800) <= 0.48, r["symbol_rate_hz"]
        assert abs(r["pulse"]["alpha"] - 0.4) <= 0.02, r["pulse"]
        assert r["doppler"]["residual_rms_hz"] <= 1.5, r["doppler"]
        assert abs(r["doppler"]["rest_freq_hz"] - rest[i]) <= 1.0, (r["doppler"]["rest_freq_hz"], rest[i])
        # The recording fades (3 dB RMS about the model), the simulation does not: medians agree only roughly.
        p50, p50_ref = r["cn0_dbhz_p10_p50_p90"][1], c["cn0_dbhz_p10_p50_p90"][1]
        assert abs(p50 - p50_ref) <= 2.0, (p50, p50_ref)
        sims.append(r)
    # The level chain. The simulation renders truth; the per-block C/N0 estimator reads it with a bias b (its
    # N0 ring at 10-20 kHz sits on the channel filter's roll-off, -0.3 dB, and on the noise shape). The level fit
    # on the simulated blocks then returns the link constant that went in (191.6 dB-Hz) plus b, so the calibration,
    # fitted with the same estimator on the recording, carries the same b.
    lv = p2b.fit_level(sims, t0, row, st, ref["level"]["saw_loss_db"])
    em = sc.source("fm117").emitters
    for i, (r, ch) in enumerate(zip(sims, lv["per_channel"], strict=True)):
        est = np.array([np.nan if v is None else v for v in r["series"]["cn0_dbhz"]], dtype=np.float64)
        true = em[i].truth(t0 + np.arange(len(est)) + 0.5, 1.0)  # 1 s blocks; FM117 rises 6 s into the segment
        d = est[np.round(true.t - t0 - 0.5).astype(int)] - true.cn0_dbhz
        b = float(np.median(d[np.isfinite(d)]))  # measured: +0.63 dB (S-5), +0.74 dB (S-8)
        assert abs(b) < 1.0, b
        assert abs(ch["link_const_dbhz"] - ref["level"]["link_const_dbhz"] - b) <= 0.05, (ch, b)
        assert ch["residual_rms_db"] <= 1.0, ch
