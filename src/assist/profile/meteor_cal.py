"""P10: METEOR level calibration from the waterfall days (perfect horizon).

Per row of each non-overlapping METEOR pass: SNR in 137.9 +- 50 kHz = (signal power in the band) / (noise power in
the band), the noise being the same bins' baseline from rows 5-15 min outside the pass. Rows where pixels saturate
(255, i.e. +25 dB over the file's calibration mean) use a fit of the LRPT spectrum (raised cosine, 72 kS/s,
alpha 0.6, shifted by the predicted Doppler) to the unsaturated bins instead of the plain sum.
Model: SNR_dB = C + G(off-nadir) + G(90 - el) - FSPL(R, f) with the QFH polynomial at both ends (plan §6.2), fitted
as 10*log10(1 + SNR) to the measured excess. C = P_tx + 30 - 10*log10(1000 k T_sys B) [dB]; only C is identifiable
from SNR, so P_tx is reported for an assumed T_sys (T_ant + receiver chain, §4.3).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import least_squares

from assist import waterfall as wf
from assist.profile.cache import centred_lut
from assist.profile.dsp import rc_psd
from assist.profile.passes import Pass, find_passes, geometry

if TYPE_CHECKING:
    from assist.profile.rawio import RawWaterfall
    from assist.profile.station import Station

QFH_COEFFS_DB = (3.369124522, 1.975440866e-2, -7.573492347e-4, -3.624644995e-6)  # plan §6.2
F_METEOR = 137.9e6
HALF_BW = 50e3
R_EARTH = 6_371_000.0
K_BOLTZ = 1.380649e-23


def qfh_gain_db(x_deg: np.ndarray) -> np.ndarray:
    """QFH gain (dBi) vs off-boresight angle; clamped to the 90 deg value outside 0..90 (plan §6.2)."""
    x = np.asarray(x_deg, dtype=np.float64)
    c0, c1, c2, c3 = QFH_COEFFS_DB
    g = c0 + c1 * x + c2 * x**2 + c3 * x**3
    g90 = c0 + c1 * 90 + c2 * 90**2 + c3 * 90**3
    return np.where((x < 0) | (x > 90), g90, g)


def fspl_db(range_m: np.ndarray, f_hz: float) -> np.ndarray:
    return 20 * np.log10(4 * np.pi * np.asarray(range_m) * f_hz / 299_792_458.0)


def off_nadir_deg(el_deg: np.ndarray, range_m: np.ndarray, station_alt_m: float) -> np.ndarray:
    """Angle at the satellite between nadir and the station (spherical Earth)."""
    rs = R_EARTH + station_alt_m
    el = np.radians(el_deg)
    r_sat = np.sqrt(rs**2 + range_m**2 + 2 * rs * range_m * np.sin(el))
    return np.degrees(np.arcsin(np.clip(rs * np.cos(el) / r_sat, -1, 1)))


def model_snr_shape_db(el_deg: np.ndarray, range_m: np.ndarray, station_alt_m: float) -> np.ndarray:
    """SNR_dB - C: transmit pattern + receive pattern - FSPL."""
    return (
        qfh_gain_db(off_nadir_deg(el_deg, range_m, station_alt_m))
        + qfh_gain_db(90 - el_deg)
        - fspl_db(range_m, F_METEOR)
    )


def band_power_fraction(rs: float = 72_000.0, alpha: float = 0.6, half_bw: float = HALF_BW) -> float:
    f = np.linspace(-rs, rs, 20001)
    p = rc_psd(f, rs, alpha)
    return float(p[np.abs(f) <= half_bw].sum() / p.sum())


@dataclass
class PassData:
    norad: int
    name: str
    p: Pass
    t: np.ndarray
    snr_lin: np.ndarray
    saturated: np.ndarray
    el: np.ndarray
    az: np.ndarray
    range_m: np.ndarray


def non_overlapping(passes_by_sat: dict[int, list[Pass]]) -> list[Pass]:
    allp = [p for ps in passes_by_sat.values() for p in ps]
    return [p for p in allp if not any(q is not p and q.aos < p.los and p.aos < q.los for q in allp)]


def measure_pass(raw: RawWaterfall, row: dict[str, str], p: Pass, station: Station) -> PassData | None:
    lut = centred_lut()
    f = raw.freqs()
    band = np.flatnonzero(np.abs(f - F_METEOR) <= HALF_BW)
    i0, i1 = raw.row_at(p.aos), raw.row_at(p.los) + 1
    pad0, pad1 = 15 * 60, 5 * 60
    a, b = max(0, i0 - pad0), min(raw.n_rows, i1 + pad0)
    if i1 - i0 < 30 or a >= i0 or b <= i1:
        return None
    px = np.asarray(raw.pixels[a:b][:, band])
    lin = 10 ** (lut[px] / 10)
    idx = np.arange(a, b)
    out_rows = ((idx < i0 - pad1) | (idx >= i1 + pad1)) & ((idx >= i0 - pad0) & (idx < i1 + pad0))
    base = np.median(lin[out_rows], axis=0)
    in_pass = (idx >= i0) & (idx < i1)
    t = np.asarray(raw.row_time(idx[in_pass]), dtype=np.float64)
    g = geometry(row, station, t)
    lin_p, px_p = lin[in_pass], px[in_pass]
    snr = (lin_p - base).sum(axis=1) / base.sum()
    sat = (px_p == 255).any(axis=1)
    if sat.any():
        snr[sat] = shape_fit_snr(lin_p[sat], px_p[sat], base, f[band], g.doppler_hz(F_METEOR)[sat])
    return PassData(int(row["NORAD_CAT_ID"]), row["OBJECT_NAME"], p, t, snr, sat, g.el_deg, g.az_deg, g.range_m)


def shape_fit_snr(
    lin: np.ndarray, px: np.ndarray, base: np.ndarray, f_hz: np.ndarray, doppler_hz: np.ndarray
) -> np.ndarray:
    """Band SNR from a least-squares fit of the LRPT spectral shape to the unsaturated bins of each row."""
    out = np.empty(len(lin))
    for i in range(len(lin)):
        s = rc_psd(f_hz - F_METEOR - doppler_hz[i], 72_000.0, 0.6) * base  # expected excess shape (noise units)
        u = px[i] < 255
        a = float(np.dot(lin[i, u] - base[u], s[u]) / np.dot(s[u], s[u]))
        out[i] = a * s.sum() / base.sum()
    return out


def edge_offsets(pd: PassData, smooth_rows: int = 10, level_db: float = 0.0) -> tuple[float, float]:
    """(rise - AOS, LOS - fall) in s: when the smoothed SNR first/last exceeds `level_db`, vs the geometric horizon.

    Positive values mean the signal appears later / disappears earlier than the horizon crossing (obstructions,
    low gain at the horizon, or element timing). Only meaningful for passes that are not clipped by the file.
    """
    snr_db = 10 * np.log10(np.maximum(np.convolve(pd.snr_lin, np.ones(smooth_rows) / smooth_rows, "same"), 1e-3))
    above = np.flatnonzero(snr_db > level_db)
    if len(above) == 0:
        return float("nan"), float("nan")
    return float(pd.t[above[0]] - pd.p.aos), float(pd.p.los - pd.t[above[-1]])


def fit_c(passes: list[PassData], station: Station, min_el: float = 3.0) -> dict[str, float]:
    el = np.concatenate([p.el for p in passes])
    rng = np.concatenate([p.range_m for p in passes])
    snr = np.concatenate([p.snr_lin for p in passes])
    ok = el >= min_el
    shape = model_snr_shape_db(el[ok], rng[ok], station.alt_m)
    meas = 10 * np.log10(np.maximum(1 + snr[ok], 1e-3))

    def f(v: np.ndarray) -> np.ndarray:
        return 10 * np.log10(1 + 10 ** ((v[0] + shape) / 10)) - meas

    r = least_squares(f, [150.0])
    return {"C_db": float(r.x[0]), "rows": int(ok.sum()), "residual_rms_db": float(np.sqrt(np.mean(r.fun**2)))}


def residual_table(passes: list[PassData], c_db: float, station: Station, min_el: float = 3.0) -> dict[str, list[Any]]:
    """Residual (measured - model, in dB of SNR) binned by elevation (5 deg) and azimuth (30 deg)."""
    el = np.concatenate([p.el for p in passes])
    az = np.concatenate([p.az for p in passes])
    rng = np.concatenate([p.range_m for p in passes])
    snr = np.concatenate([p.snr_lin for p in passes])
    ok = (el >= min_el) & (snr > 0.25)
    model = c_db + model_snr_shape_db(el[ok], rng[ok], station.alt_m)
    res = 10 * np.log10(snr[ok]) - model
    out: dict[str, list[Any]] = {"el_bins": [], "az_bins": []}
    for lo in range(0, 90, 5):
        m = (el[ok] >= lo) & (el[ok] < lo + 5)
        if m.sum() >= 10:
            out["el_bins"].append([lo, int(m.sum()), float(np.median(res[m])), float(np.std(res[m]))])
    for lo in range(0, 360, 30):
        m = (az[ok] >= lo) & (az[ok] < lo + 30)
        if m.sum() >= 10:
            out["az_bins"].append([lo, int(m.sum()), float(np.median(res[m])), float(np.std(res[m]))])
    return out


def t_sys_k(
    t_ant_k: float, saw_loss_db: float, lna_nf_db: float = 1.0, lna_gain_db: float = 20.0, tuner_nf_db: float = 3.5
) -> float:
    """System temperature referred to the antenna (plan §4.3)."""
    l_saw = 10 ** (saw_loss_db / 10)
    t_lna = 290 * (10 ** (lna_nf_db / 10) - 1)
    t_tun = 290 * (10 ** (tuner_nf_db / 10) - 1)
    return float(t_ant_k + (l_saw - 1) * 290 + l_saw * t_lna + l_saw * t_tun / 10 ** (lna_gain_db / 10))


def p_tx_dbw(c_db: float, t_sys: float, band_hz: float = 2 * HALF_BW, band_fraction: float = 1.0) -> float:
    """P_tx (dBW) from C for a given T_sys; band_fraction corrects for signal power outside the measured band."""
    return float(c_db + 10 * np.log10(K_BOLTZ * t_sys * band_hz) - 10 * np.log10(band_fraction))


def collect(
    raws: list[RawWaterfall], rows: dict[int, dict[str, str]], station: Station, norads: tuple[int, ...]
) -> list[PassData]:
    out: list[PassData] = []
    for raw in raws:
        t0, t1 = float(raw.row_time(0)), float(raw.row_time(raw.n_rows - 1))
        by_sat = {n: find_passes(rows[n], station, t0, t1) for n in norads}
        keep = non_overlapping(by_sat)
        for p in keep:
            pd = measure_pass(raw, rows[p.norad], p, station)
            if pd is not None:
                out.append(pd)
    return out


__all__ = ["F_METEOR", "wf"]
