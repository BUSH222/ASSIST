"""Link budget (plan §6.2) and system noise temperature (§4.3).

P_rx[dBm] = P_tx[dBW] + 30 + G_tx(off-nadir) - FSPL(R, f) + G_rx(90 - el) - L_horizon(az, el) - L_extra
C/N0[dB-Hz] = P_rx[dBm] - 30 - 10 log10(k T_sys)

Calibrated levels are stored as a "link constant": the C/N0 the link would give at 0 dB total path gain
(G_tx + G_rx - FSPL - losses = 0). It does not depend on T_sys, so a calibrated SNR stays right whatever T_ant is
assumed; P_tx = link_const + 10 log10(k T_sys).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

C_M_S = 299_792_458.0
K_BOLTZ = 1.380649e-23
QFH_COEFFS_DB = (3.369124522, 1.975440866e-2, -7.573492347e-4, -3.624644995e-6)  # plan §6.2, dBi vs degrees


def poly_gain_db(x_deg: np.ndarray | float, coeffs_db: Sequence[float], max_deg: float = 90.0) -> np.ndarray:
    """Polynomial gain (dBi) vs off-boresight angle; outside 0..max_deg it is clamped to the max_deg value."""
    x = np.asarray(x_deg, dtype=np.float64)
    c = np.asarray(coeffs_db, dtype=np.float64)
    g = np.polynomial.polynomial.polyval(x, c)
    return np.where((x < 0) | (x > max_deg), np.polynomial.polynomial.polyval(max_deg, c), g)


def qfh_gain_db(x_deg: np.ndarray | float) -> np.ndarray:
    """QFH gain (dBi) vs off-boresight angle; clamped to the 90 deg value outside 0..90 (plan §6.2)."""
    x = np.asarray(x_deg, dtype=np.float64)
    c0, c1, c2, c3 = QFH_COEFFS_DB
    g = c0 + c1 * x + c2 * x**2 + c3 * x**3
    g90 = c0 + c1 * 90 + c2 * 90**2 + c3 * 90**3
    return np.where((x < 0) | (x > 90), g90, g)


def cardioid_gain_db(x_deg: np.ndarray | float, n: float, peak_dbi: float = 0.0) -> np.ndarray:
    """peak + 10 n log10((1 + cos x) / 2), floored 60 dB below the peak."""
    x = np.radians(np.asarray(x_deg, dtype=np.float64))
    rel = np.maximum((1 + np.cos(x)) / 2, 1e-30)
    return np.maximum(peak_dbi + 10 * n * np.log10(rel), peak_dbi - 60.0)


@dataclass(frozen=True)
class Pattern:
    """Antenna gain (dBi) vs off-boresight angle (deg): `poly` (QFH by default), `cardioid` or `isotropic`."""

    type: str = "poly"
    coeffs_db: tuple[float, ...] = QFH_COEFFS_DB
    n: float = 1.0
    peak_dbi: float = 0.0
    gain_dbi: float = 0.0

    def gain_db(self, x_deg: np.ndarray | float) -> np.ndarray:
        if self.type == "poly":
            if tuple(self.coeffs_db) == QFH_COEFFS_DB:
                return qfh_gain_db(x_deg)
            return poly_gain_db(x_deg, self.coeffs_db)
        if self.type == "cardioid":
            return cardioid_gain_db(x_deg, self.n, self.peak_dbi)
        if self.type == "isotropic":
            return np.full(np.shape(x_deg), self.gain_dbi, dtype=np.float64)
        raise ValueError(f"unknown pattern type {self.type!r}")


def fspl_db(range_m: np.ndarray | float, f_hz: float | np.ndarray) -> np.ndarray:
    return 20 * np.log10(4 * np.pi * np.asarray(range_m) * f_hz / C_M_S)


def t_sys_k(
    t_ant_k: float, saw_loss_db: float, lna_nf_db: float = 1.0, lna_gain_db: float = 20.0, tuner_nf_db: float = 3.5
) -> float:
    """System temperature referred to the antenna (plan §4.3)."""
    l_saw = 10 ** (saw_loss_db / 10)
    t_lna = 290 * (10 ** (lna_nf_db / 10) - 1)
    t_tun = 290 * (10 ** (tuner_nf_db / 10) - 1)
    return float(t_ant_k + (l_saw - 1) * 290 + l_saw * t_lna + l_saw * t_tun / 10 ** (lna_gain_db / 10))


def kt_dbw_hz(t_sys: float) -> float:
    """10 log10(k T_sys): noise density in dBW/Hz."""
    return float(10 * np.log10(K_BOLTZ * t_sys))


class HorizonModel:
    """Horizon loss L_horizon(az, el) in dB. A perfect horizon (zero) until M10 builds the map (plan §6.2)."""

    def loss_db(self, az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
        return np.zeros(np.broadcast_shapes(np.shape(az_deg), np.shape(el_deg)))


def p_rx_dbm(
    p_tx_dbw: float,
    g_tx_db: np.ndarray,
    range_m: np.ndarray,
    f_hz: float | np.ndarray,
    g_rx_db: np.ndarray,
    l_horizon_db: np.ndarray | float = 0.0,
    l_extra_db: float = 0.0,
) -> np.ndarray:
    return p_tx_dbw + 30 + g_tx_db - fspl_db(range_m, f_hz) + g_rx_db - l_horizon_db - l_extra_db


def link_const_from_snr_c(c_db: float, band_hz: float, band_fraction: float = 1.0) -> float:
    """Link constant from an in-band SNR constant C (SNR = C + path gain), as fitted in M1 P10.

    SNR = band_fraction * P_rx / (k T_sys B), so C/N0 = SNR * B / band_fraction.
    """
    return float(c_db + 10 * np.log10(band_hz) - 10 * np.log10(band_fraction))


def p_tx_from_link_const(link_const_dbhz: float, t_sys: float) -> float:
    return link_const_dbhz + kt_dbw_hz(t_sys)


def p_tx_for_cn0_zenith(
    cn0_dbhz: float,
    height_m: float,
    f_hz: float,
    tx: Pattern,
    rx: Pattern,
    t_sys: float,
    l_extra_db: float = 0.0,
) -> float:
    """P_tx (dBW) that gives cn0_dbhz for an overhead pass at height_m above the station (plan §6.2)."""
    gain = float(tx.gain_db(0.0)) + float(rx.gain_db(0.0)) - float(fspl_db(height_m, f_hz)) - l_extra_db
    return cn0_dbhz - gain + kt_dbw_hz(t_sys)
