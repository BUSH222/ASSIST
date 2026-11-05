"""P3: noise PSD, DC spike, diurnal level curve and noise regimes.

Reference shape: the two .raw days (raw rtl_tcp stream, the same path ASSIST emulates and that M9 compares
against). Per day: per-bin median over per-minute means. Where a day's waterfall is clipped at its -5 dB floor
(SAW notch below ~136.05 MHz) the recording's shape is spliced in. Persistent signals (strong VDL, birdies, DC)
are interpolated out, then the shape is smoothed. 0 dB = median over the flat passband.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from assist import waterfall as wf

if TYPE_CHECKING:
    from collections.abc import Sequence

FLAT_MHZ = (136.2, 137.8)
VDL_MASK_MHZ = ((136.905, 136.945), (136.962, 136.988))  # near-continuous VDL2 ground stations
FLOOR_DB = float(wf.pixel_to_db(np.array([0]), centre=True)[0])  # lowest value a .raw pixel can say
CENTRE = wf.FFT_SIZE // 2


def flat_mask(f_mhz: np.ndarray) -> np.ndarray:
    m = (f_mhz > FLAT_MHZ[0]) & (f_mhz < FLAT_MHZ[1])
    for lo, hi in VDL_MASK_MHZ:
        m &= ~((f_mhz > lo) & (f_mhz < hi))
    return m


def normalise(shape_db: np.ndarray, f_mhz: np.ndarray) -> np.ndarray:
    return shape_db - np.median(shape_db[flat_mask(f_mhz)])


def interpolate_masked(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace masked bins by linear interpolation from the unmasked ones."""
    out = y.copy()
    idx = np.arange(len(y))
    out[mask] = np.interp(idx[mask], idx[~mask], y[~mask])
    return out


def dc_spike(shape_db: np.ndarray) -> dict[str, float]:
    """Excess of the centre bin (and +-1) over the local median of bins 5..40 away."""
    ref = float(np.median(np.concatenate([shape_db[CENTRE - 40 : CENTRE - 4], shape_db[CENTRE + 5 : CENTRE + 41]])))
    return {
        "centre_bin_excess_db": float(shape_db[CENTRE] - ref),
        "plus_minus_1_bin_excess_db": float((shape_db[CENTRE - 1] + shape_db[CENTRE + 1]) / 2 - ref),
    }


def dc_amplitude_counts(centre_excess_db: float, noise_sigma_counts: float, fs: float = wf.FS) -> float:
    """|DC| in counts from the centre-bin excess over noise (Nuttall window, tone exactly on the bin).

    Tone power in the bin = A^2 * (sum w)^2 / N^2-normalised; noise in the bin = N0 * ENBW. Equivalently the
    excess ratio r = 10^(dB/10) - 1 equals A^2 / (sigma_c^2 * ENBW_bins / N) with sigma_c^2 the complex noise
    power (2 * sigma^2 per component).
    """
    w = wf.window()
    enbw_bins = len(w) * np.sum(w**2) / np.sum(w) ** 2
    r = 10 ** (centre_excess_db / 10) - 1
    sigma_c2 = 2 * noise_sigma_counts**2
    return float(np.sqrt(max(r, 0.0) * sigma_c2 * enbw_bins / len(w)))


def day_shape(minute_mean_db: np.ndarray, keep_minutes: np.ndarray | None = None) -> np.ndarray:
    m = minute_mean_db if keep_minutes is None else minute_mean_db[keep_minutes]
    return np.median(m, axis=0)


def combined_shape(
    day_shapes: Sequence[np.ndarray], wav_shape: np.ndarray | None, f_mhz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(raw combined shape normalised, clipped-bin mask). Days are averaged; clipped bins come from the WAV."""
    norm = [normalise(s, f_mhz) for s in day_shapes]
    clipped = np.zeros(len(f_mhz), bool)
    for s in day_shapes:
        clipped |= s <= FLOOR_DB + 0.5
    shape = np.mean(norm, axis=0)
    if wav_shape is not None and clipped.any():
        w = normalise(wav_shape, f_mhz)
        # Match the WAV to the days just outside each clipped region (40 bins).
        edge = np.zeros_like(clipped)
        for s, e in _runs(clipped):
            edge[e : min(len(edge), e + 40)] = True
            edge[max(0, s - 40) : s] = True
        edge &= ~clipped
        off = float(np.median((shape - w)[edge])) if edge.any() else 0.0
        shape[clipped] = w[clipped] + off
    return shape, clipped


def _runs(b: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], b.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist(), strict=True))


def smooth_shape(shape_db: np.ndarray, f_mhz: np.ndarray, extra_mask: np.ndarray | None = None) -> np.ndarray:
    """Persistent signals and DC interpolated out, then a 31-bin median + 15-bin mean (~4.5 kHz)."""
    mask = np.zeros(len(f_mhz), bool)
    for lo, hi in VDL_MASK_MHZ:
        mask |= (f_mhz > lo) & (f_mhz < hi)
    mask[CENTRE - 3 : CENTRE + 4] = True
    if extra_mask is not None:
        mask |= extra_mask
    y = interpolate_masked(shape_db, mask)
    y = median_filter(y, size=31, mode="nearest")
    return uniform_filter1d(y, size=15, mode="nearest")


def hourly_curve(row_level_db: np.ndarray, row_times_unix: np.ndarray) -> np.ndarray:
    """Median band level per UTC hour (24 values, NaN where empty), relative to the day's median."""
    hours = ((row_times_unix % 86400) // 3600).astype(int)
    out = np.full(24, np.nan)
    for h in range(24):
        sel = hours == h
        if sel.any():
            out[h] = np.median(row_level_db[sel])
    return out - np.nanmedian(row_level_db)


def regimes(minute_level_db: np.ndarray) -> dict[str, Any]:
    """Two-level description of the noise floor: split per-minute band levels at the largest density gap."""
    lv = np.sort(minute_level_db)
    lo, hi = np.percentile(lv, [5, 95])
    grid = np.linspace(lo, hi, 200)
    hist = np.histogram(lv, bins=grid)[0].astype(float)
    hist = uniform_filter1d(hist, 7)
    # Threshold: the lowest density between the two largest modes.
    peaks = np.argsort(hist)[::-1]
    p1 = int(peaks[0])
    p2 = next((int(p) for p in peaks[1:] if abs(int(p) - p1) > 20), p1)
    a, b = sorted((p1, p2))
    cut = grid[a + int(np.argmin(hist[a : b + 1]))] if b > a else float(np.median(lv))
    loud = minute_level_db > cut
    segs = _runs(loud)
    durs = [e - s for s, e in segs]
    return {
        "threshold_db": float(cut),
        "quiet_level_db": float(np.median(minute_level_db[~loud])) if (~loud).any() else float("nan"),
        "loud_level_db": float(np.median(minute_level_db[loud])) if loud.any() else float("nan"),
        "loud_fraction": float(loud.mean()),
        "loud_episodes": len(segs),
        "loud_episode_minutes_p10_p50_p90": [float(x) for x in np.percentile(durs, [10, 50, 90])] if durs else [],
    }
