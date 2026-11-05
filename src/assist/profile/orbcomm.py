"""P2b: ORBCOMM reference from the annotated segment of the recording.

Per channel: channelize to 50 kS/s; carrier track from the squared signal. Measured on the recording: like MSK,
+-90 deg DPSK puts two equal lines in x^2 at 2 (df -+ rs/4), i.e. at df -+ 1200 Hz in carrier units, and the
carrier is their midpoint (it matches the spectral centroid; a single x^2 line would be off by 1200 Hz).
Doppler check against the predicted range rate (rest frequency, time
offset and residual); symbol rate from the envelope's cyclostationary line; pulse shape from a raised-cosine fit
to the Doppler-corrected PSD; C/N0 per second.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import welch

from assist.emitters.link import fspl_db, p_tx_from_link_const, qfh_gain_db, t_sys_k
from assist.orbits.geometry import C_M_S, geometry
from assist.profile.dsp import channelize, rc_psd

if TYPE_CHECKING:
    from assist.profile.wavio import Recording
    from assist.station import Station

RATE = 50_000
DECIM = 48  # 2.4 MS/s -> 50 kS/s


def _line(y: np.ndarray, power: int, nfft: int, lo_hz: float, hi_hz: float) -> tuple[float, float]:
    """(carrier offset Hz, peak-to-median dB) of the x^power line searched in [lo_hz, hi_hz] (carrier units)."""
    y = y.astype(np.complex128)
    y = y / (np.abs(y) + 1e-12)  # constant modulus: keeps the phase only
    z = np.abs(np.fft.fft(y**power * np.hanning(len(y)), nfft)) ** 2
    fr = np.fft.fftfreq(nfft, 1 / RATE)
    sel = np.flatnonzero((fr >= power * lo_hz) & (fr <= power * hi_hz))
    k = int(sel[np.argmax(z[sel])])
    a, b, c = (np.log(z[(k + i) % nfft] + 1e-30) for i in (-1, 0, 1))
    d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    return float((fr[k] + d * RATE / nfft) / power), float(10 * np.log10(z[k] / np.median(z)))


def _pair(y: np.ndarray, nfft: int, centre_hz: float, half_sep_hz: float, search_hz: float) -> tuple[float, float]:
    """(carrier offset Hz, weaker line's peak-to-median dB): midpoint of the x^2 lines at centre -+ half_sep."""
    lo, q_lo = _line(y, 2, nfft, centre_hz - half_sep_hz - search_hz, centre_hz - half_sep_hz + search_hz)
    hi, q_hi = _line(y, 2, nfft, centre_hz + half_sep_hz - search_hz, centre_hz + half_sep_hz + search_hz)
    return (lo + hi) / 2, min(q_lo, q_hi)


def carrier_track(
    x: np.ndarray, expected_hz: np.ndarray, rs: float = 4800.0, block: int = RATE, search_hz: float = 300.0
) -> dict[str, np.ndarray]:
    """Per block: carrier offset (Hz, midpoint of the x^2 line pair) and the weaker line's peak-to-median (dB).

    The search is limited to expected +- search_hz around each line: x^2 has further, weaker symbol-rate lines.
    """
    n_blk = len(x) // block
    t, f, q = [], [], []
    for b in range(n_blk):
        off, pk = _pair(x[b * block : (b + 1) * block], 4 * block, float(expected_hz[b]), rs / 4, search_hz)
        t.append((b + 0.5) * block / RATE)
        f.append(off)
        q.append(pk)
    return {"t": np.array(t), "offset_hz": np.array(f), "peak_db": np.array(q)}


def anchored_expectation(
    x: np.ndarray, t0_unix: float, row: dict[str, str], station: Station, f_centre_hz: float, block: int = RATE
) -> np.ndarray:
    """Expected per-block offset: the carrier measured over the strongest 10 s, moved by the predicted Doppler."""
    n_blk = len(x) // block
    pw = np.array([np.mean(np.abs(x[b * block : (b + 1) * block]) ** 2) for b in range(n_blk)])
    span = 10
    k = int(np.argmax(np.convolve(pw, np.ones(span), mode="valid")))
    seg = x[k * block : (k + span) * block]
    # Coarse centre = spectral centroid of the signal above the noise; it picks the carrier, not a sideline
    # (x^2 also has lines at 2 df +- rs that follow the same Doppler curve).
    fr, p = welch(seg, fs=RATE, nperseg=4096, return_onesided=False, scaling="density")
    sig = np.clip(p - np.median(p[(np.abs(fr) > 12_000)]), 0, None) * (np.abs(fr) < 9000)
    c0 = float(np.sum(fr * sig) / np.sum(sig))
    off, _ = _pair(seg, 4 * span * block, c0, 1200.0, 600.0)
    t_blk = t0_unix + (np.arange(n_blk) + 0.5) * block / RATE
    d = -f_centre_hz * geometry(row, station, t_blk).range_rate_m_s / C_M_S
    d_anchor = float(np.mean(d[k : k + span]))
    return off + d - d_anchor


def fit_doppler(
    t_unix: np.ndarray, f_obs_hz: np.ndarray, row: dict[str, str], station: Station, f_nominal: float
) -> dict[str, Any]:
    """f_obs = f_rest * (1 - rr(t + dt) / c): fit f_rest and dt; report residuals (Hz)."""

    def model(dt: float) -> np.ndarray:
        rr = geometry(row, station, t_unix + dt).range_rate_m_s
        return -f_nominal * rr / C_M_S

    best = None
    for dt in np.arange(-10.0, 10.01, 0.25):
        d = model(dt)
        f_rest = float(np.median(f_obs_hz - d))
        res = f_obs_hz - f_rest - d
        rms = float(np.sqrt(np.mean(res**2)))
        if best is None or rms < best[0]:
            best = (rms, dt, f_rest, res)
    assert best is not None
    rms, dt, f_rest, res = best
    d0 = model(0.0)
    f0 = float(np.median(f_obs_hz - d0))
    res0 = f_obs_hz - f0 - d0
    slope = float(np.polyfit(t_unix - t_unix[0], res0, 1)[0])
    return {
        "rest_freq_hz": f0,
        "residual_rms_hz": float(np.sqrt(np.mean(res0**2))),
        "residual_max_abs_hz": float(np.max(np.abs(res0))),
        "residual_slope_hz_per_s": slope,
        "doppler_span_hz": [float(d0.min()), float(d0.max())],
        "best_time_offset_s": float(dt),
        "rest_freq_at_best_offset_hz": f_rest,
        "residual_rms_at_best_offset_hz": rms,
    }


def symbol_rate(x_corr: np.ndarray, guess_hz: float = 4800.0, span_hz: float = 300.0) -> float:
    """Symbol rate from the spectral line of |x|^2 near `guess_hz`."""
    e = np.abs(x_corr) ** 2
    e = e - e.mean()
    nfft = 1 << int(np.ceil(np.log2(len(e))))
    z = np.abs(np.fft.rfft(e * np.hanning(len(e)), nfft)) ** 2
    fr = np.fft.rfftfreq(nfft, 1 / RATE)
    sel = np.flatnonzero(np.abs(fr - guess_hz) < span_hz)
    k = sel[np.argmax(z[sel])]
    a, b, c = np.log(z[k - 1]), np.log(z[k]), np.log(z[k + 1])
    return float(fr[k] + 0.5 * (a - c) / (a - 2 * b + c) * RATE / nfft)


def derotate(x: np.ndarray, t_track: np.ndarray, f_track: np.ndarray) -> np.ndarray:
    """Remove a smooth carrier-offset track (Hz at block centres) by integrating it per sample."""
    t = np.arange(len(x)) / RATE
    f = np.interp(t, t_track, f_track)
    phase = 2 * np.pi * np.cumsum(f) / RATE
    return x * np.exp(-1j * phase).astype(np.complex64)


def fit_pulse(x_corr: np.ndarray, rs: float) -> dict[str, Any]:
    """Fit A * RC(f; rs, alpha) + N to the averaged PSD of the Doppler-corrected signal (|f| < 12 kHz)."""
    fr, p = welch(x_corr, fs=RATE, nperseg=4096, return_onesided=False, scaling="density")
    order = np.argsort(fr)
    fr, p = fr[order], p[order]
    sel = np.abs(fr) < 12_000
    f, y = fr[sel], p[sel]
    noise0 = float(np.median(p[(np.abs(fr) > 8000) & (np.abs(fr) < 20000)]))

    def resid(v: np.ndarray) -> np.ndarray:
        a, alpha, n, df = v
        return 10 * np.log10(a * rc_psd(f - df, rs, alpha) + n) - 10 * np.log10(y)

    v0 = np.array([float(y.max()), 0.4, noise0, 0.0])
    r = least_squares(resid, v0, bounds=([0, 0.05, 0, -200], [np.inf, 1.0, np.inf, 200]))
    a, alpha, n, df = r.x
    sig = np.clip(y - n, 0, None)
    c = np.cumsum(sig) / sig.sum()
    obw = float(f[np.searchsorted(c, 0.995)] - f[np.searchsorted(c, 0.005)])
    try:
        cov = np.linalg.inv(r.jac.T @ r.jac) * np.mean(r.fun**2)
        alpha_err = float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        alpha_err = float("nan")
    return {
        "alpha": float(alpha),
        "alpha_stderr": alpha_err,
        "fit_rms_db": float(np.sqrt(np.mean(r.fun**2))),
        "occupied_bw_99_hz": obw,
        "centre_offset_hz": float(df),
        "psd_peak_over_noise_db": float(10 * np.log10(a / n + 1)),
    }


def cn0_per_block(x_corr: np.ndarray, rs: float, alpha: float, block: int = RATE) -> np.ndarray:
    """C/N0 (dB-Hz) per block: in-band power minus noise, over the noise density from |f| in 10-20 kHz."""
    out = []
    half = (1 + alpha) * rs / 2
    for b in range(len(x_corr) // block):
        y = x_corr[b * block : (b + 1) * block]
        fr, p = welch(y, fs=RATE, nperseg=2048, return_onesided=False, scaling="density")
        n0 = float(np.median(p[(np.abs(fr) > 10_000) & (np.abs(fr) < 20_000)]))
        inb = np.abs(fr) <= half
        c = float((p[inb] - n0).sum() * (fr[1] - fr[0]))
        out.append(10 * np.log10(c / n0) if c > 0 else np.nan)
    return np.array(out)


def doppler_fit_for(
    x: np.ndarray, t0_unix: float, f_centre_hz: float, row: dict[str, str], station: Station
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(carrier track, Doppler fit) of channelized samples `x` against one satellite's elements."""
    tr = carrier_track(x, anchored_expectation(x, t0_unix, row, station, f_centre_hz))
    good = tr["peak_db"] > 15
    if good.sum() < 10:
        return tr, {"blocks_with_carrier": int(good.sum())}
    dop = fit_doppler(t0_unix + tr["t"][good], f_centre_hz + tr["offset_hz"][good], row, station, f_centre_hz)
    dop["blocks_with_carrier"] = int(good.sum())
    return tr, dop


def analyse(
    x: np.ndarray, t0_unix: float, f_centre_hz: float, row: dict[str, str], station: Station
) -> dict[str, Any]:
    """Full P2b analysis of channelized samples `x` (50 kS/s around f_centre_hz) for one satellite."""
    tr, dop = doppler_fit_for(x, t0_unix, f_centre_hz, row, station)
    t_dense = np.arange(0, len(x) / RATE + 1, 1.0)
    rr = geometry(row, station, t0_unix + t_dense).range_rate_m_s
    f_model = float(dop["rest_freq_hz"]) * (1 - rr / C_M_S) - f_centre_hz
    xc = derotate(x, t_dense, f_model)
    rs = symbol_rate(xc)
    pulse = fit_pulse(xc, rs)
    cn0 = cn0_per_block(xc, rs, float(pulse["alpha"]))
    el = geometry(row, station, t0_unix + tr["t"]).el_deg
    rr_blk = geometry(row, station, t0_unix + tr["t"]).range_rate_m_s
    resid = f_centre_hz + tr["offset_hz"] - float(dop["rest_freq_hz"]) * (1 - rr_blk / C_M_S)
    fr, p = welch(xc, fs=RATE, nperseg=4096, return_onesided=False, scaling="density")
    o = np.argsort(fr)
    fr, p = fr[o], p[o]
    keep = np.abs(fr) < 12_000
    a_fit = float(pulse["psd_peak_over_noise_db"])
    noise_d = float(np.median(p[(np.abs(fr) > 8000) & (np.abs(fr) < 20000)]))
    model = noise_d * (
        1 + (10 ** (a_fit / 10) - 1) * rc_psd(fr - float(pulse["centre_offset_hz"]), rs, float(pulse["alpha"]))
    )
    return {
        "channel_centre_hz": f_centre_hz,
        "blocks": len(tr["t"]),
        "doppler": dop,
        "symbol_rate_hz": rs,
        "pulse": pulse,
        "cn0_dbhz_p10_p50_p90": [float(v) for v in np.nanpercentile(cn0, [10, 50, 90])],
        "series": {
            "t_s": tr["t"].tolist(),
            "offset_hz": tr["offset_hz"].tolist(),
            "peak_db": tr["peak_db"].tolist(),
            "el_deg": el.tolist(),
            "cn0_dbhz": [None if not np.isfinite(v) else float(v) for v in cn0],
            "residual_hz": resid.tolist(),
        },
        "psd": {
            "f_hz": fr[keep][::4].tolist(),
            "psd_db": (10 * np.log10(p[keep][::4] / noise_d)).tolist(),
            "model_db": (10 * np.log10(model[keep][::4] / noise_d)).tolist(),
        },
    }


def fit_level(
    channels: list[dict[str, Any]],
    t0_unix: float,
    row: dict[str, str],
    station: Station,
    saw_loss_db: float,
    max_residual_hz: float = 5.0,
) -> dict[str, Any]:
    """Transmit level from the per-block C/N0, with the simulator's link model (plan §6.2).

    Path gain = QFH(off-nadir) + QFH(90 - el) - FSPL(range, f): the QFH pattern at both ends, perfect horizon.
    Per channel, the link constant is the median of C/N0 - path gain over blocks where the carrier is tracked
    (|Doppler residual| < max_residual_hz), the satellite is above the horizon and C/N0 is finite. The median
    keeps the deep fade 90 s into the segment from biasing it. P_tx = link constant + 10 log10(k T_sys).
    """
    per = []
    for ch in channels:
        s = ch["series"]
        t = t0_unix + np.asarray(s["t_s"], dtype=np.float64)
        cn0 = np.array([np.nan if v is None else v for v in s["cn0_dbhz"]], dtype=np.float64)
        resid = np.asarray(s["residual_hz"], dtype=np.float64)
        g = geometry(row, station, t)
        f_hz = float(ch["doppler"]["rest_freq_hz"])
        path = qfh_gain_db(g.off_nadir_deg) + qfh_gain_db(90.0 - g.el_deg) - fspl_db(g.range_m, f_hz)
        ok = np.isfinite(cn0) & (np.abs(resid) < max_residual_hz) & (g.el_deg >= 0)
        k = cn0[ok] - path[ok]
        med = float(np.median(k))
        per.append(
            {
                "channel_hz": f_hz,
                "link_const_dbhz": med,
                "blocks_used": int(ok.sum()),
                "residual_rms_db": float(np.sqrt(np.mean((k - med) ** 2))),
                "residual_p10_p90_db": [float(v) for v in np.percentile(k - med, [10, 90])],
                "el_deg_range": [float(g.el_deg[ok].min()), float(g.el_deg[ok].max())],
                "off_nadir_deg_range": [float(g.off_nadir_deg[ok].min()), float(g.off_nadir_deg[ok].max())],
            }
        )
    link_const = float(np.mean([c["link_const_dbhz"] for c in per]))
    return {
        "method": fit_level.__doc__,
        "per_channel": per,
        "link_const_dbhz": link_const,
        "saw_loss_db": saw_loss_db,
        "p_tx_dbw_for_t_ant": {
            str(t_ant): {
                "t_sys_k": t_sys_k(t_ant, saw_loss_db),
                "p_tx_dbw": p_tx_from_link_const(link_const, t_sys_k(t_ant, saw_loss_db)),
            }
            for t_ant in (300, 500, 1000)
        },
    }


def channelize_segment(rec: Recording, n0: int, n1: int, f_centre_hz: float) -> np.ndarray:
    return channelize(rec.iq, n0, n1, rec.fs, f_centre_hz - 137.0e6, DECIM)


# ORBCOMM channels seen on the waterfall days (MHz): subscriber downlinks and the gateway at 137.56.
CHANNELS_SEEN_MHZ = (137.2875, 137.46, 137.56, 137.6625, 137.7125, 137.7375, 137.8)
