"""Birdies (persistent narrow lines).

A bin is a birdie if it exceeds the median of its neighbours (5..40 bins away on each side) by more than
`threshold_db` in more than `min_fraction` of the per-minute mean spectra. Per-minute means (60 rows) are used
instead of single rows: a persistent line is just as persistent there, and the per-bin noise drops from ~0.36 dB
to ~0.05 dB, so smaller lines can be found. Adjacent bins merge into one birdie.
"""

import numpy as np
from scipy.ndimage import median_filter

from assist import waterfall as wf

CENTRE = wf.FFT_SIZE // 2


def local_excess(spectra_db: np.ndarray) -> np.ndarray:
    """Excess of each bin over a local median that skips the bin's own +-4 neighbours (81-bin footprint)."""
    fp = np.ones(81, bool)
    fp[36:45] = False
    ref = median_filter(spectra_db, footprint=fp[None, :] if spectra_db.ndim == 2 else fp, mode="nearest")
    return spectra_db - ref


def find_birdies(
    minute_mean_db: np.ndarray, f_hz: np.ndarray, threshold_db: float = 0.5, min_fraction: float = 0.9
) -> list[dict[str, float]]:
    exc = local_excess(minute_mean_db)
    frac = (exc > threshold_db).mean(axis=0)
    hit = frac > min_fraction
    hit[CENTRE - 3 : CENTRE + 4] = False  # DC is described separately
    out: list[dict[str, float]] = []
    d = np.diff(np.concatenate([[0], hit.astype(np.int8), [0]]))
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True):
        k = s + int(np.argmax(np.median(exc[:, s:e], axis=0)))
        out.append(
            {
                "freq_hz": float(f_hz[k]),
                "width_bins": int(e - s),
                "excess_db_median": float(np.median(exc[:, k])),
                "fraction": float(frac[k]),
            }
        )
    return out
