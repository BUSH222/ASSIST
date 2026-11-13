"""P14 (M7): terrestrial calibration, and the derivation of `bands/vhf137/terrestrial_default.yaml`.

**Measure** (`assist profile terrestrial`, needs the recording and `out/profile/events.csv`; writes
`terrestrial.json` into the profile):
- VDL bursts on 136.925 and 136.975 MHz in the recording's clean part (after the local interference, before the
  ORBCOMM pass): each channel is cut out by FFT (+-10 kHz, 50 kS/s), bursts are found on the 0.2 ms power (8 dB on,
  3 dB off over the 20th percentile) and measured in absolute ADC counts: mean power minus the channel floor, over
  the noise density ASSIST renders at that frequency (the profile's noise shape and sigma). That makes the level
  the same C/N0 ASSIST uses (`cn0_dbhz` of a ground emitter), and the same bursts clip the same way. Detections
  shorter than 10 ms are fragments (and impulses), not bursts.
- The modulation of a channel comes from the mean spectrum of its strongest cluster: VDL Mode 2 D8PSK is ~11 kHz
  wide at -10 dB, VDL Mode 4 GFSK (constant envelope, +-2.4 kHz deviation) only ~4 kHz (M7: 136.925 MHz is GFSK).
- Tight level clusters (>= 10 % of the bursts within +-1 dB, twice as dense as the 1-4 dB flanks) are fixed
  transmitters (ground stations); the rest are aircraft (spread levels). A cluster
  whose bursts put many samples on the ADC rails is clipped: its true level comes from the rail fraction (the
  background rail fraction subtracted) by simulating bursts of the channel's modulation with Gaussian noise through
  the rails.
- From the event list of the waterfall days: AM transmissions per channel (the 8.33 kHz grid; birdie channels are
  labelled birdie there) and the per-event carrier level, and the 138 MHz carrier levels, from the peak excess in
  1 s rows: C/N0 = 10 log10(10^(X/10) - 1) + 10 log10(ENBW), with the capture script's Nuttall window.

**Derive** (`derive`, from the committed profile files only, so CI can check the band file against it): presets'
parameters for this station. Hourly curves (24 UTC values) are smoothed hard (mean plus the first daily harmonic)
and scaled so the rate at the recording's mid-time is the recording's.
"""

import csv
import json
import math
from functools import cache
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from assist import waterfall as wf
from assist.profile.dsp import hysteresis, runs

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class Reader(Protocol):
    """What the measurement reads: the recording (profile.wavio.Recording), or a rendered stream in tests."""

    fs: int
    start_unix: float

    def iq(self, n0: int, n: int, /) -> np.ndarray: ...

    def u8(self, n0: int, n: int, /) -> np.ndarray: ...


FO = 50_000  # channel rate
HALF_HZ = 10_000  # channel half width (raised-cosine edge of 1 kHz)
EDGE_HZ = 1_000
SMOOTH = 10  # 0.2 ms power smoothing
ON_DB, OFF_DB = 8.0, 3.0
MIN_BURST_S = 0.010
CLUSTER_HALF_DB = 3.0
CLUSTER_MIN_SHARE = 0.10
CLUSTER_CONTRAST = 2.0  # a fixed transmitter's level cluster is >= 2x as dense as its +-1..4 dB flanks
GFSK_MAX_WIDTH_HZ = 7000.0  # -10 dB width: VDL4 GFSK ~4 kHz, VDL2 D8PSK ~11 kHz
RAIL = 126.5  # |u - 127.5| at the recording's rails (1 and 254)
VDL_MHZ = (136.925, 136.975)
D8PSK_RS = 10_500
AIR_MIN_CN0 = 50.0  # aircraft bursts: measured from here up (the detection limit)
BIRDIE_WIDE_BINS = 10  # wider birdies are rendered as noise_band


# --- measurement -------------------------------------------------------------------------------------------------


def _mask(f: np.ndarray) -> np.ndarray:
    a = np.abs(f)
    lo, hi = HALF_HZ - EDGE_HZ / 2, HALF_HZ + EDGE_HZ / 2
    return np.where(a <= lo, 1.0, np.where(a >= hi, 0.0, 0.5 * (1 + np.cos(np.pi * (a - lo) / EDGE_HZ))))


def enbw_hz() -> float:
    """Noise bandwidth of the channel mask."""
    f = np.arange(-2 * HALF_HZ, 2 * HALF_HZ + 1, dtype=np.float64)
    return float(np.sum(_mask(f) ** 2))


def channelize_fft(rec: Reader, t0_s: float, t1_s: float, centres_hz: list[float]) -> dict[float, np.ndarray]:
    """Channels (complex64, ADC counts, FO samples/s) of seconds [t0_s, t1_s) of the recording, by FFT per second with
    50 ms of overlap on each side."""
    fs = rec.fs
    pad = int(0.05 * fs)
    ln = fs + 2 * pad
    f = np.fft.fftfreq(ln, 1 / fs)
    n_s = int(t1_s - t0_s)
    out = {c: np.empty(n_s * FO, np.complex64) for c in centres_hz}
    m_out = ln * FO // fs
    m0 = pad * FO // fs
    for i in range(n_s):
        x = rec.iq(int((t0_s + i) * fs) - pad, ln) * np.float32(127.5)
        big = np.fft.fft(x)
        for c in centres_hz:
            rel = f - (c - 137.0e6)
            sel = np.flatnonzero(np.abs(rel) < FO / 2)
            y = np.zeros(m_out, np.complex128)
            y[np.round(rel[sel] * ln / fs).astype(np.int64) % m_out] = big[sel] * _mask(rel[sel])
            out[c][i * FO : (i + 1) * FO] = (np.fft.ifft(y) * (m_out / ln))[m0 : m0 + FO]
    return out


def assist_n0_counts(profile_dir: Path, f_hz: float, fs: int = 2_400_000, fc_hz: float = 137.0e6) -> float:
    """Noise density (ADC counts^2 per Hz, complex) that ASSIST renders at f_hz with this profile: 2 sigma^2 |H(f)|^2
    / (fs sum |h|^2), whatever T_sys is."""
    from assist.engine.frontend import design_fir

    prof = json.loads((profile_dir / "profile.json").read_text())
    shape = np.load(profile_dir / prof["noise"]["psd_file"]).astype(np.float64)
    sigma = float(prof["adc"]["noise_sigma_mad_counts"])
    h = design_fir(shape)
    k = np.arange(-(len(h) // 2), len(h) // 2 + 1)
    hf = np.sum(h * np.exp(-2j * np.pi * (f_hz - fc_hz) * k / fs))
    return float(2 * sigma**2 * abs(hf) ** 2 / (fs * np.sum(np.abs(h) ** 2)))


def bursts(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """([start, end) sample pairs, mean power minus floor (counts^2) of each burst, floor power), bursts >= 10 ms."""
    pw = np.abs(y.astype(np.complex128)) ** 2
    sm = np.convolve(pw, np.ones(SMOOTH) / SMOOTH, mode="same")
    exc = 10 * np.log10(np.maximum(sm, 1e-30) / np.percentile(sm, 20))
    on = hysteresis(exc, ON_DB, OFF_DB)
    ev = runs(on)
    quiet = ~np.convolve(on, np.ones(200), mode="same").astype(bool)
    floor = float(pw[quiet].mean())
    ev = ev[(ev[:, 1] - ev[:, 0]) >= MIN_BURST_S * FO]
    p = np.array([pw[a:b].mean() for a, b in ev]) - floor
    return ev, p, floor


def rail_fraction(rec: Reader, t0_s: float, ev: np.ndarray) -> np.ndarray:
    """Fraction of I and Q values on the rails (1/254 or beyond) inside each burst."""
    out = np.empty(len(ev))
    for i, (a, b) in enumerate(ev):
        n0 = int((t0_s + a / FO) * rec.fs)
        u = rec.u8(n0, int((b - a) / FO * rec.fs)).astype(np.int16)
        out[i] = np.mean(np.abs(u - 127.5) >= RAIL)
    return out


def background_rail_fraction(rec: Reader, t0_s: float, ev: np.ndarray, n_total: int) -> float:
    """Rail fraction in 200 stretches of 5 ms away from bursts."""
    busy = np.zeros(n_total, bool)
    for a, b in ev:
        busy[max(0, a - 500) : b + 500] = True
    free = np.flatnonzero(~busy[: n_total - 250])
    pick = free[np.linspace(0, len(free) - 1, 200).astype(np.int64)]
    vals = []
    for a in pick:
        n0 = int((t0_s + a / FO) * rec.fs)
        u = rec.u8(n0, int(0.005 * rec.fs)).astype(np.int16)
        vals.append(np.mean(np.abs(u - 127.5) >= RAIL))
    return float(np.mean(vals))


def mean_spectrum_width_db10(x: np.ndarray, rate: float) -> float:
    """-10 dB width (Hz) of the mean of 10.24 ms Hann spectra of x (the resolution of the 50 kS/s measurement)."""
    seg = round(0.01024 * rate)
    m = len(x) // seg
    p = (np.abs(np.fft.fft(x[: m * seg].reshape(m, seg) * np.hanning(seg), axis=1)) ** 2).mean(axis=0)
    f = np.fft.fftfreq(seg, 1 / rate)
    return float(np.ptp(f[p >= p.max() / 10]))


def fit_gfsk_transitions(width_hz: float) -> float:
    """The gfsk transition probability whose -10 dB spectrum width is width_hz (bisection; width falls with it)."""
    from assist.backend import dsp
    from assist.emitters.schedules import Event
    from assist.rng import KeyedRng
    from assist.signals.base import Keys, make_signal

    def width(pt: float) -> float:
        sig = make_signal("gfsk", {"transition_p": pt}, Keys(KeyedRng(0), "fit", "gfsk"), dsp)
        rate = 4 * sig.rate_step()
        ev = Event(0.0, 4.0, {"bits": 4 * sig.rate_step(), "duration_s": 4.0})
        return mean_spectrum_width_db10(sig.render_events([ev], 0, 4 * rate, rate).astype(np.complex128), rate)

    a, b = 0.5, 0.95
    if width(a) <= width_hz:
        return a
    for _ in range(20):
        m = (a + b) / 2
        a, b = (m, b) if width(m) > width_hz else (a, m)
    return round((a + b) / 2, 3)


@cache
def _burst_samples(kind: str, transition_p: float = 0.5) -> np.ndarray:
    """A long noiseless burst of the modulation (unit power, ramps cut off), sampled finely."""
    from assist.backend import dsp
    from assist.emitters.schedules import Event
    from assist.rng import KeyedRng
    from assist.signals.base import Keys, make_signal

    params = {"transition_p": transition_p} if kind == "gfsk" else {}
    sig = make_signal(kind, params, Keys(KeyedRng(0), "rails", kind), dsp)
    step = sig.rate_step()
    rate = 8 * step
    if kind == "gfsk":
        ev = Event(0.0, 2.0, {"bits": 2 * step, "duration_s": 2.0})
    else:
        ev = Event(0.0, 2.0, {"symbols": 2 * step, "duration_s": 2.0})
    x = sig.render_events([ev], 0, 2 * rate, rate).astype(np.complex128)[rate // 10 : -rate // 10]
    return x / math.sqrt(np.mean(np.abs(x) ** 2))


def simulated_rail_fraction(power_counts2: float, sigma: float, kind: str = "d8psk", tp: float = 0.5) -> float:
    """Rail fraction of bursts of `kind` (d8psk or gfsk with transition probability tp) at this mean power plus
    Gaussian noise of sigma per component."""
    x = _burst_samples(kind, tp) * math.sqrt(power_counts2)
    # A fixed seed of the fitting tool, not of any scenario: the fit lands in the profile, never in a render.
    n = np.random.default_rng(0).normal(0, sigma, (2, len(x)))
    v = np.concatenate([x.real + n[0], x.imag + n[1]])
    return float(np.mean(np.abs(v) >= RAIL))


def invert_rails(
    frac: float, sigma: float, lo_db: float, kind: str = "d8psk", tp: float = 0.5, hi_db: float = 60.0
) -> float:
    """Mean power (counts^2, in dB) whose simulated rail fraction is `frac` (bisection on dB)."""
    a, b = lo_db, hi_db
    for _ in range(40):
        m = (a + b) / 2
        if simulated_rail_fraction(10 ** (m / 10), sigma, kind, tp) < frac:
            a = m
        else:
            b = m
    return (a + b) / 2


def _pct(x: np.ndarray, q: list[float]) -> list[float]:
    return [round(float(v), 3) for v in np.percentile(x, q)] if len(x) else []


def vdl_channel(
    rec: Reader, t0_s: float, y: np.ndarray, n0: float, sigma: float, log: Callable[[str], None]
) -> dict[str, Any]:
    """Bursts of one channel split into fixed transmitters (clusters) and the rest (aircraft)."""
    ev, p, floor = bursts(y)
    span_s = len(y) / FO
    ok = p > 0
    ev, p = ev[ok], p[ok]
    cn0 = 10 * np.log10(p / n0)
    dur = (ev[:, 1] - ev[:, 0]) / FO
    rails = rail_fraction(rec, t0_s, ev)
    bg = background_rail_fraction(rec, t0_s, ev, len(y))
    # clusters: 0.5 dB histogram peaks holding >= 10 % of the bursts within +-1 dB
    edges = np.arange(math.floor(cn0.min()), math.ceil(cn0.max()) + 0.5, 0.5)
    h, _ = np.histogram(cn0, edges)
    centres = (edges[:-1] + edges[1:]) / 2
    found: list[float] = []
    for i in np.argsort(h)[::-1]:
        c = float(centres[i])
        near = np.abs(cn0 - c) <= 1.0
        if near.sum() < CLUSTER_MIN_SHARE * len(cn0):
            break
        flank = (np.abs(cn0 - c) > 1.0) & (np.abs(cn0 - c) <= 4.0)  # 6 dB of flanks vs 2 dB of peak
        narrow = near.sum() / 2.0 >= CLUSTER_CONTRAST * flank.sum() / 6.0
        if narrow and all(abs(c - f) > 2 * CLUSTER_HALF_DB for f in found):
            found.append(float(np.median(cn0[near])))
    kind, tp, w10 = "d8psk", 0.5, 0.0
    if found:  # the modulation, from the mean spectrum of the strongest cluster (by level)
        top = np.flatnonzero(np.abs(cn0 - max(found)) <= 1.0)[:300]
        acc = np.zeros(512)
        for j in top:
            m = (ev[j, 0] + ev[j, 1]) // 2
            acc += np.abs(np.fft.fft(y[m - 256 : m + 256].astype(np.complex128) * np.hanning(512))) ** 2
        f = np.fft.fftfreq(512, 1 / FO)
        w10 = float(np.ptp(f[acc >= acc.max() / 10]))
        kind = "gfsk" if w10 < GFSK_MAX_WIDTH_HZ else "d8psk"
        if kind == "gfsk":
            tp = fit_gfsk_transitions(w10)
        log(f"  modulation {kind} (-10 dB width {w10:.0f} Hz{f', transition_p {tp}' if kind == 'gfsk' else ''})")
    taken = np.zeros(len(cn0), bool)
    clusters = []
    for c in sorted(found):
        sel = (np.abs(cn0 - c) <= CLUSTER_HALF_DB) & ~taken
        taken |= sel
        r_obs = float(np.median(rails[sel]))
        r_sig = max(0.0, (r_obs - bg) / (1 - bg))
        measured = float(np.median(cn0[sel]))
        corrected = measured
        if r_sig > 0.002:
            p_db = invert_rails(r_sig, sigma, 10 * math.log10(float(np.median(p[sel]))) - 3, kind, tp)
            corrected = p_db - 10 * math.log10(n0)
        clusters.append(
            {
                "cn0_dbhz_measured": round(measured, 3),
                "cn0_dbhz": round(max(corrected, measured), 3),
                "clipped": r_sig > 0.002,
                "rail_fraction_p50": round(r_obs, 5),
                "bursts": int(sel.sum()),
                "rate_per_s": round(float(sel.sum()) / span_s, 4),
                "duration_ms_p10_p25_p50_p75_p90": _pct(dur[sel] * 1e3, [10, 25, 50, 75, 90]),
            }
        )
        log(f"  cluster {measured:.1f} dB-Hz -> {corrected:.1f} ({sel.sum()} bursts, rails {r_obs:.3f})")
    rest = ~taken & (cn0 >= AIR_MIN_CN0)
    air = cn0[rest]
    return {
        "modulation": kind,
        "spectrum_width_10db_hz": round(w10, 1),
        **({"gfsk_transition_p": tp} if kind == "gfsk" else {}),
        "floor_counts2_per_hz": round(floor / enbw_hz(), 6),
        "assist_n0_counts2_per_hz": round(n0, 6),
        "floor_over_assist_db": round(10 * math.log10(floor / enbw_hz() / n0), 3),
        "background_rail_fraction": round(bg, 5),
        "bursts": len(cn0),
        "fixed": clusters,
        "spread": {
            "bursts": int(rest.sum()),
            "rate_per_s": round(float(rest.sum()) / span_s, 4),
            "cn0_dbhz_p10_p16_p50_p84_p90": _pct(air, [10, 16, 50, 84, 90]),
            "cn0_dbhz_min_max": _pct(air, [0, 100]),
            "duration_ms_p10_p25_p50_p75_p90": _pct(dur[rest] * 1e3, [10, 25, 50, 75, 90]),
        },
    }


def nuttall_enbw_hz() -> float:
    w = wf.window()
    return float(wf.FS * np.sum(w**2) / np.sum(w) ** 2)


def excess_to_cn0(x_db: np.ndarray | float, width_hz: float | None = None) -> np.ndarray:
    """C/N0 of a line (or a band of FWHM width_hz) whose peak stands x_db above the noise in the waterfall."""
    b = nuttall_enbw_hz() if width_hz is None else 1.0645 * width_hz  # Gaussian band: peak density C / (1.0645 W)
    lin = np.maximum(10 ** (np.asarray(x_db, dtype=np.float64) / 10) - 1, 1e-6)
    return 10 * np.log10(lin * b)


def events_stats(events_csv: Path, n_days: int = 2) -> dict[str, Any]:
    """AM per channel and the 138 MHz carrier from the event list of the waterfall days."""
    am: dict[str, list[float]] = {}
    am_dur: list[float] = []
    c138: list[float] = []
    with events_csv.open() as f:
        for r in csv.DictReader(f):
            lab = r["auto_label"]
            if lab == "am":
                dur = (int(r["row_end"]) - int(r["row_start"])) * wf.ROW_SECONDS
                if dur > 60:  # long carriers, not voice
                    continue
                ch = str(round((float(r["f_lo_hz"]) + float(r["f_hi_hz"])) / 2))  # channel centre, Hz
                am.setdefault(ch, []).append(float(r["peak_db"]))
                am_dur.append(dur)
            elif lab == "carrier_138":
                c138.append(float(r["peak_db"]))
    allam = np.concatenate([np.array(v) for v in am.values()])
    am_cn0 = excess_to_cn0(allam)
    chans = {k: round(len(v) / n_days, 2) for k, v in sorted(am.items(), key=lambda kv: -len(kv[1]))}
    c_cn0 = excess_to_cn0(np.array(c138))
    return {
        "am": {
            "events_per_day_by_channel_hz": chans,
            "carrier_cn0_dbhz_p10_p50_p90": _pct(am_cn0, [10, 50, 90]),
            "events": len(allam),
        },
        "carrier_138": {"cn0_dbhz_p10_p50_p90": _pct(c_cn0, [10, 50, 90]), "events": len(c138)},
        "enbw_hz": round(nuttall_enbw_hz(), 3),
    }


def measure(
    rec: Reader,
    profile_dir: Path,
    events_csv: Path,
    t0_s: float,
    t1_s: float,
    log: Callable[[str], None] = print,
    cache: Path | None = None,
) -> dict[str, Any]:
    prof = json.loads((profile_dir / "profile.json").read_text())
    sigma = float(prof["adc"]["noise_sigma_mad_counts"])
    centres = [f * 1e6 for f in VDL_MHZ]
    chans: dict[float, np.ndarray]
    cf = cache / f"terrestrial_channels_{int(t0_s)}_{int(t1_s)}.npz" if cache else None
    if cf is not None and cf.exists():
        d = np.load(cf)
        chans = {c: d[f"arr_{i}"] for i, c in enumerate(centres)}
    else:
        log(f"channelizing {t1_s - t0_s:.0f} s of the recording ...")
        chans = channelize_fft(rec, t0_s, t1_s, centres)
        if cf is not None:
            cf.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cf, *[chans[c] for c in centres])  # arr_0, arr_1
    vdl = {}
    for c in centres:
        log(f"VDL {c / 1e6:.3f} MHz")
        vdl[f"{c / 1e6:.3f}"] = vdl_channel(rec, t0_s, chans[c], assist_n0_counts(profile_dir, c), sigma, log)
    mid_h = ((rec.start_unix + (t0_s + t1_s) / 2) % 86400) / 3600
    return {
        "version": 1,
        "recording_window_s": [t0_s, t1_s],
        "recording_window_mid_utc_h": round(mid_h, 4),
        "method": "P14 (M7): VDL bursts in absolute ADC counts vs ASSIST's noise; AM/138 from the days' event list",
        "vdl": vdl,
        **events_stats(events_csv),
    }


# --- derivation of the band's terrestrial scenario ---------------------------------------------------------------


def smooth_daily(curve: list[float]) -> Callable[[float], float]:
    """Mean plus the first daily harmonic of 24 hourly values (index h covers [h, h + 1) UTC); a function of the UTC
    hour, floored at 10 % of the mean."""
    y = np.asarray(curve, dtype=np.float64)
    h = np.arange(24) + 0.5
    w = 2 * np.pi * h / 24
    a0 = y.mean()
    a1 = 2 * np.mean(y * np.cos(w))
    b1 = 2 * np.mean(y * np.sin(w))

    def f(t_h: float) -> float:
        x = 2 * np.pi * t_h / 24
        return float(max(a0 + a1 * np.cos(x) + b1 * np.sin(x), 0.1 * a0))

    return f


def hourly(shape: Callable[[float], float], rate_at_ref_per_h: float, ref_h: float) -> list[float]:
    """24 hourly rates following `shape`, equal to rate_at_ref_per_h at UTC hour ref_h."""
    k = rate_at_ref_per_h / shape(ref_h)
    return [round(k * shape(h + 0.5), 3) for h in range(24)]


def daily(shape: Callable[[float], float], per_day: float) -> list[float]:
    """24 hourly rates following `shape` that sum to per_day."""
    v = np.array([shape(h + 0.5) for h in range(24)])
    return [round(float(x), 4) for x in v / v.sum() * per_day]


def dead_time_rate(kept_per_s: float, busy_s: float) -> float:
    """Poisson rate whose kept rate is kept_per_s with a dead time busy_s (min_gap_s + burst)."""
    return kept_per_s / max(1e-9, 1 - kept_per_s * busy_s)


def _burst_ms(d: list[float]) -> list[float]:
    """burst_ms range from duration percentiles (p25 .. p90), at least 0.5 ms wide."""
    lo, hi = float(d[1]), float(d[4])
    if hi - lo < 0.5:
        lo = hi - 0.5
    return [round(lo, 2), round(hi, 2)]


GROUND_MIN_GAP_S = 0.05
AM_MIN_PER_DAY = 10.0
AM_SPILL_RATIO = 4.0


def derive(profile_dir: Path) -> dict[str, Any]:
    """The `sources` of bands/vhf137/terrestrial_default.yaml from the profile (profile.json, birdies.json,
    terrestrial.json)."""
    prof = json.loads((profile_dir / "profile.json").read_text())
    ter = json.loads((profile_dir / "terrestrial.json").read_text())
    bird = json.loads((profile_dir / "birdies.json").read_text())
    occ = prof["occupancy"]
    ref_h = float(ter["recording_window_mid_utc_h"])
    traffic = smooth_daily(occ["vdl"]["vdl_136.975"]["events_per_hour_utc"])  # aircraft follow 136.975's curve
    sources: list[dict[str, Any]] = []
    # Aircraft: one level distribution (from the channel with the most aircraft bursts), rates per channel.
    air = max((v["spread"] for v in ter["vdl"].values()), key=lambda sp: sp["bursts"])
    for ch, v in ter["vdl"].items():
        f_hz = float(ch) * 1e6
        own = smooth_daily(occ["vdl"][f"vdl_{ch}"]["events_per_hour_utc"])
        kind = v["modulation"]
        sig_params: dict[str, Any] = {"transition_p": v["gfsk_transition_p"]} if kind == "gfsk" else {}
        for i, c in enumerate(v["fixed"]):
            d = c["duration_ms_p10_p25_p50_p75_p90"]
            busy = GROUND_MIN_GAP_S + d[2] / 1000
            rate = dead_time_rate(c["rate_per_s"], busy) * 3600
            sources.append(
                {
                    "id": f"vdl-{ch}-{'abcdefgh'[i]}",
                    "preset": "vdl_ground",
                    "freq_hz": f_hz,
                    "level": {"cn0_dbhz": round(c["cn0_dbhz"], 2)},
                    "schedule": {"rate_per_hour": hourly(own, rate, ref_h), "min_gap_s": GROUND_MIN_GAP_S},
                    "signal": {"kind": kind, "params": {"burst_ms": _burst_ms(d), **sig_params}},
                }
            )
        sp = v["spread"]
        if sp["bursts"] >= 10:
            p = air["cn0_dbhz_p10_p16_p50_p84_p90"]
            lo, hi = air["cn0_dbhz_min_max"]
            med = p[2]
            sources.append(
                {
                    "id": f"vdl-air-{ch}",
                    "preset": "vdl_aircraft",
                    "freq_hz": f_hz,
                    "level": {"cn0_dbhz": round(med, 2)},
                    "schedule": {"rate_per_hour": hourly(traffic, sp["rate_per_s"] * 3600, ref_h)},
                    "signal": {
                        "kind": kind,
                        "params": {"burst_ms": _burst_ms(air["duration_ms_p10_p25_p50_p75_p90"])},
                    },
                    "per_event": {
                        "offset_hz": [-2000.0, 2000.0],
                        "level_db": {
                            "dist": "normal",
                            "sigma_db": round((p[3] - p[1]) / 2, 2),
                            "min_db": round(lo - med, 2),
                            "max_db": round(hi - med, 2),
                        },
                    },
                }
            )
    # AM voice per channel (the days' event list), shaped by the smoothed AM curve
    am = ter["am"]
    am_shape = smooth_daily(occ["am"]["events_per_hour_utc"])
    a10, a50, a90 = am["carrier_cn0_dbhz_p10_p50_p90"]
    sig = round((a90 - a50) / 1.2816, 2)
    dur = occ["am"]["duration_s"]
    chans = {int(k): float(v) for k, v in am["events_per_day_by_channel_hz"].items()}
    for ch, per_day in am["events_per_day_by_channel_hz"].items():
        if per_day < AM_MIN_PER_DAY:
            continue
        # spill: a busier channel one 8.33 kHz step away with >= 4x the events (the same transmissions)
        if any(abs(int(ch) - c) < 8400 and v >= AM_SPILL_RATIO * per_day for c, v in chans.items()):
            continue
        sources.append(
            {
                "id": f"am-{int(ch) / 1e6:.4f}",
                "preset": "atc_am",
                "freq_hz": float(ch),
                "level": {"cn0_dbhz": round(a50, 2)},
                "schedule": {
                    "rate_per_hour": daily(am_shape, per_day),
                    "duration": {"median_s": round(math.exp(dur["mu"]), 3), "sigma": round(dur["sigma"], 3)},
                },
                "per_event": {
                    "level_db": {"sigma_db": sig, "min_db": round(a10 - a50 - sig, 2), "max_db": round(3 * sig, 2)}
                },
            }
        )
    # 138 MHz carrier
    b = occ["band138"]
    c10, c50, c90 = ter["carrier_138"]["cn0_dbhz_p10_p50_p90"]
    csig = round((c90 - c50) / 1.2816, 2)
    sources.append(
        {
            "id": "carrier-138",
            "preset": "carrier_138",
            "freq_hz": round(float(b["freq_mhz_p10_p50_p90"][1]) * 1e6, 3),
            "level": {"cn0_dbhz": round(c50, 2)},
            "schedule": {
                "rate_per_hour": daily(smooth_daily(b["events_per_hour_utc"]), b["events_per_day"]),
                "duration": {
                    "median_s": round(math.exp(b["duration_s"]["mu"]), 3),
                    "sigma": round(b["duration_s"]["sigma"], 3),
                },
            },
            "per_event": {
                "level_db": {"sigma_db": csig, "min_db": round(c10 - c50 - csig, 2), "max_db": round(3 * csig, 2)}
            },
        }
    )
    # USB humps (the waterfall days, the rtl_tcp path)
    days = [v["comb"] for k, v in prof["humps"].items() if k.endswith(".raw")]
    rate = float(np.mean([d["episodes_per_hour"] for d in days]))
    med = float(np.mean([d["duration_s_p10_p50_p90"][1] for d in days]))
    hsig = float(
        np.mean([math.log(d["duration_s_p10_p50_p90"][2] / d["duration_s_p10_p50_p90"][1]) / 1.2816 for d in days])
    )
    amps = [a for d in days for a in d["amplitude_db_p10_p50_p90"]]
    wander = float(np.mean([d["wander_hz_s_p50_p90"][0] for d in days])) / 0.6745  # median |v| of normal(0, s)
    sources.append(
        {
            "id": "usb-humps",
            "preset": "usb_humps",
            "schedule": {
                "rate_per_hour": round(rate, 3),
                "duration": {"median_s": round(med, 2), "sigma": round(hsig, 3)},
            },
            "signal": {
                "params": {
                    "spacing_hz": [round(d["spacing_hz_p10_p50_p90"][1], 1) for d in days],
                    "amplitude_db": [round(min(amps), 3), round(max(amps), 3)],
                    "wander_hz_s": round(wander, 1),
                }
            },
        }
    )
    # Birdies: CW tones (wide ones: noise_band)
    ems = []
    bin_hz = wf.FS / wf.FFT_SIZE
    for i, bd in enumerate(bird["birdies"]):
        f = round(float(bd["freq_hz"]), 3)
        if int(bd["width_bins"]) >= BIRDIE_WIDE_BINS:
            w = round(int(bd["width_bins"]) * bin_hz, 1)
            sigcfg: dict[str, Any] = {"kind": "noise_band", "params": {"width_hz": w}}
            cn0 = float(excess_to_cn0(float(bd["excess_db"]), w))
        else:
            sigcfg = {"kind": "cw"}
            cn0 = float(excess_to_cn0(float(bd["excess_db"])))
        ems.append({"id": f"b{i + 1:02d}", "signal": sigcfg, "freq_hz": f, "level": {"cn0_dbhz": round(cn0, 2)}})
    sources.append({"id": "birdies", "preset": "birdies", "emitters": ems})
    return {"sources": sources}


HEADER = """\
# Terrestrial environment of the vhf137 station (plan §6.5, M7): VDL ground stations and aircraft bursts (136.925 MHz:
# VDL Mode 4 GFSK; 136.975 MHz: VDL Mode 2 D8PSK), airband AM voice, the 138.1104 MHz carrier, USB noise humps and
# birdies. Include it in a scenario:
#   include: [bands/vhf137/terrestrial_default.yaml]
# Generated by `assist profile terrestrial --band` from profiles/vhf137 (profile.json, birdies.json,
# terrestrial.json); a test checks that it matches, so change the derivation (assist.profile.terrestrial.derive)
# rather than this file. Override in a scenario by source id: sources merge by id, e.g.
#   sources: [{id: usb-humps, comb: {enabled: false}}]      # no humps (`comb` is the emitter)
# Levels are C/N0 against k T_sys at each frequency (the SNR a measurement there reads), from the recording's ADC
# counts (VDL; clipped stations corrected through their rail fraction) or the waterfall days (AM, 138 MHz, birdies).
# Hourly rates (24 UTC values) are smoothed to the mean plus the first daily harmonic.
"""


def band_yaml(profile_dir: Path) -> str:
    """The text of terrestrial_default.yaml."""
    import yaml

    body = str(yaml.safe_dump(derive(profile_dir), sort_keys=False, width=119, default_flow_style=None))
    return HEADER + body
