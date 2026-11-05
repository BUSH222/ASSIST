"""P8: noise humps ("USB humps") and wandering lines.

Rows minus the hour's baseline spectrum, known narrow signals masked, smoothed over 10 rows x 51 bins and
reduced to 10-row steps x 8-bin columns (1.17 kHz).

Humps: on the recording they are a periodic comb of +-0.2..0.5 dB ripples across the whole band that drifts (the
chevrons). Per step the detrended residual over 136.1-137.84 MHz is Fourier-analysed across frequency: the dominant
spacing (15-150 kHz), its ripple amplitude (dB, peak) and phase. Steps with amplitude above a threshold form
episodes; the phase drift gives the comb's wander speed.

Wandering lines: narrow peaks linked into tracks across steps (max 600 Hz/s); only tracks lasting >= 15 min count.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.ndimage import uniform_filter, uniform_filter1d
from scipy.signal import find_peaks

from assist import waterfall as wf
from assist.profile.cache import centred_lut

if TYPE_CHECKING:
    from assist.profile.rawio import RawWaterfall

STEP_ROWS = 10
COL_BINS = 8
COL_HZ = COL_BINS * wf.FS / wf.FFT_SIZE
MASK_MHZ = (
    (136.88, 137.0),  # VDL
    (137.84, 137.96),  # METEOR
    (137.282, 137.298),  # ORBCOMM channels seen on the days (+-8 kHz)
    (137.452, 137.468),
    (137.548, 137.572),
    (137.652, 137.668),
    (137.70, 137.75),
    (137.792, 137.808),
)
BAND_MHZ = (136.08, 138.15)


def _mask(f_mhz: np.ndarray) -> np.ndarray:
    m = (f_mhz < BAND_MHZ[0]) | (f_mhz > BAND_MHZ[1])
    for lo, hi in MASK_MHZ:
        m |= (f_mhz > lo) & (f_mhz < hi)
    m[wf.FFT_SIZE // 2 - 4 : wf.FFT_SIZE // 2 + 5] = True
    return m


def residual_cube(rows_db: np.ndarray, baseline_db: np.ndarray, f_mhz: np.ndarray) -> np.ndarray:
    """(n_rows // STEP_ROWS, FFT_SIZE // COL_BINS) smoothed residual dB of a block of rows (one baseline)."""
    r = rows_db - baseline_db
    m = _mask(f_mhz)
    r[:, m] = 0.0
    r = uniform_filter(r, size=(STEP_ROWS, 51), mode="nearest")
    n = len(r) // STEP_ROWS * STEP_ROWS
    r = r[STEP_ROWS // 2 : n : STEP_ROWS]
    r = r.reshape(len(r), -1, COL_BINS).mean(axis=2)
    r[:, m.reshape(-1, COL_BINS).any(axis=1)] = np.nan
    return r.astype(np.float32)


def day_cube(raw: RawWaterfall, baselines: np.ndarray) -> np.ndarray:
    """Residual cube of a whole .raw day, one baseline per hour (from occupancy.hourly_baseline)."""
    lut = centred_lut()
    f = raw.freqs() / 1e6
    out = []
    for h in range(int(np.ceil(raw.n_rows / 3600))):
        db = lut[np.asarray(raw.pixels[h * 3600 : (h + 1) * 3600])]
        out.append(residual_cube(db, baselines[min(h, len(baselines) - 1)], f))
    return np.concatenate(out)


def detect(cube: np.ndarray, min_db: float = 0.1, max_db: float = 2.0, min_width_cols: int = 4) -> list[np.ndarray]:
    """Per step: array of (column, height dB, width cols) of broad excursions."""
    out = []
    for r in cube:
        x = np.nan_to_num(r, nan=0.0)
        x = x - uniform_filter1d(x, 401, mode="nearest")  # remove slow tilts (> ~470 kHz)
        pk, props = find_peaks(x, prominence=min_db, width=min_width_cols, height=(min_db, max_db))
        out.append(np.stack([pk, props["peak_heights"], props["widths"]], axis=1) if len(pk) else np.zeros((0, 3)))
    return out


def track(
    dets: list[np.ndarray], step_s: float, max_speed_hz_s: float = 600.0, min_steps: int = 6
) -> list[np.ndarray]:
    """Greedy nearest-neighbour linking; returns tracks as arrays of (step, column, height, width)."""
    max_cols = max_speed_hz_s * step_s / COL_HZ
    active: list[list[tuple[float, float, float, float]]] = []
    done: list[list[tuple[float, float, float, float]]] = []
    for s, d in enumerate(dets):
        used = np.zeros(len(d), bool)
        still: list[list[tuple[float, float, float, float]]] = []
        for tr in active:
            last = tr[-1]
            if len(d):
                dist = np.abs(d[:, 0] - last[1])
                dist[used] = np.inf
                j = int(np.argmin(dist))
                if dist[j] <= max_cols:
                    tr.append((s, d[j, 0], d[j, 1], d[j, 2]))
                    used[j] = True
                    still.append(tr)
                    continue
            done.append(tr)
        for j in np.flatnonzero(~used):
            still.append([(s, d[j, 0], d[j, 1], d[j, 2])])
        active = still
    done += active
    return [np.array(t) for t in done if len(t) >= min_steps]


def summarise(tracks: list[np.ndarray], step_s: float, hours: float, f0_hz: float) -> dict[str, Any]:
    if not tracks:
        return {"episodes": 0}
    dur = np.array([(t[-1, 0] - t[0, 0] + 1) * step_s for t in tracks])
    amp = np.array([np.median(t[:, 2]) for t in tracks])
    width = np.array([np.median(t[:, 3]) * COL_HZ for t in tracks])
    speed = np.concatenate([np.abs(np.diff(t[:, 1])) * COL_HZ / step_s for t in tracks if len(t) > 1])
    centre = np.array([np.median(t[:, 1]) * COL_HZ + f0_hz for t in tracks])
    # Comb: spacing between simultaneous track points
    by_step: dict[int, list[float]] = {}
    for t in tracks:
        for s, c, *_ in t:
            by_step.setdefault(int(s), []).append(float(c))
    sp = np.concatenate([np.diff(sorted(v)) * COL_HZ for v in by_step.values() if len(v) > 1] or [np.zeros(0)])
    hist, edges = np.histogram(sp, bins=np.arange(0, 400e3, 5e3)) if len(sp) else (np.zeros(0), np.zeros(1))
    return {
        "episodes": len(tracks),
        "episodes_per_hour": len(tracks) / hours,
        "duration_s_p10_p50_p90": [float(v) for v in np.percentile(dur, [10, 50, 90])],
        "amplitude_db_p10_p50_p90": [float(v) for v in np.percentile(amp, [10, 50, 90])],
        "width_hz_p10_p50_p90": [float(v) for v in np.percentile(width, [10, 50, 90])],
        "wander_hz_s_p50_p90": [float(v) for v in np.percentile(speed, [50, 90])] if len(speed) else [],
        "centre_mhz_p10_p50_p90": [float(v) / 1e6 for v in np.percentile(centre, [10, 50, 90])],
        "comb_spacing_mode_hz": float(edges[np.argmax(hist)] + 2.5e3) if len(sp) else None,
        "simultaneous_pairs": len(sp),
    }


COMB_MHZ = (136.1, 137.84)


def comb_series(
    cube: np.ndarray, f_col_mhz: np.ndarray, min_hz: float = 15e3, max_hz: float = 150e3
) -> dict[str, np.ndarray]:
    """Per step: dominant ripple spacing (Hz), peak amplitude (dB) and phase (rad) of the comb."""
    sel = (f_col_mhz >= COMB_MHZ[0]) & (f_col_mhz <= COMB_MHZ[1])
    x = cube[:, sel].astype(np.float64)
    n = x.shape[1]
    nfft = 8 * n
    q = np.fft.rfftfreq(nfft, d=COL_HZ)  # cycles per Hz
    band = (q >= 1 / max_hz) & (q <= 1 / min_hz)
    spacing, amp, phase = [], [], []
    idx = np.arange(n)
    win = np.hanning(n)
    for r in x:
        m = np.isfinite(r)
        if m.sum() < n // 2:
            spacing.append(np.nan), amp.append(np.nan), phase.append(np.nan)
            continue
        y = np.interp(idx, idx[m], r[m])
        y = y - uniform_filter1d(y, 201, mode="nearest")
        z = np.fft.rfft(y * win, nfft)
        k = np.flatnonzero(band)[int(np.argmax(np.abs(z[band])))]
        spacing.append(1 / q[k])
        amp.append(2 * np.abs(z[k]) / win.sum())
        phase.append(np.angle(z[k]))
    return {"spacing_hz": np.array(spacing), "amp_db": np.array(amp), "phase": np.array(phase)}


def comb_summary(cs: dict[str, np.ndarray], step_s: float, threshold_db: float = 0.1) -> dict[str, Any]:
    on = np.nan_to_num(cs["amp_db"]) > threshold_db
    d = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
    ep = [(s0, e0) for s0, e0 in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True) if e0 - s0 >= 3]
    if not ep:
        return {"fraction_on": float(on.mean()), "episodes": 0}
    dur = np.array([(e0 - s0) * step_s for s0, e0 in ep])
    sp = np.concatenate([cs["spacing_hz"][s0:e0] for s0, e0 in ep])
    am = np.concatenate([cs["amp_db"][s0:e0] for s0, e0 in ep])
    # Wander: phase drift between consecutive steps, as Hz/s of the comb position (same spacing only)
    w = []
    for s0, e0 in ep:
        ph, spc = cs["phase"][s0:e0], cs["spacing_hz"][s0:e0]
        same = np.abs(np.diff(spc)) < 0.05 * spc[1:]
        dph = np.angle(np.exp(1j * np.diff(ph)))[same]
        w.append(np.abs(dph) / (2 * np.pi) * spc[1:][same] / step_s)
    wv = np.concatenate(w) if w else np.zeros(0)
    return {
        "fraction_on": float(on.mean()),
        "episodes": len(ep),
        "episodes_per_hour": len(ep) / (len(on) * step_s / 3600),
        "duration_s_p10_p50_p90": [float(v) for v in np.percentile(dur, [10, 50, 90])],
        "spacing_hz_p10_p50_p90": [float(v) for v in np.percentile(sp, [10, 50, 90])],
        "amplitude_db_p10_p50_p90": [float(v) for v in np.percentile(am, [10, 50, 90])],
        "wander_hz_s_p50_p90": [float(v) for v in np.percentile(wv, [50, 90])] if len(wv) else [],
    }


def wandering_lines(
    tracks: list[np.ndarray], step_s: float, f_col_mhz: np.ndarray, min_s: float = 900.0
) -> list[dict[str, Any]]:
    out = []
    for t in tracks:
        dur = (t[-1, 0] - t[0, 0] + 1) * step_s
        if dur < min_s:
            continue
        f = f_col_mhz[t[:, 1].astype(int)]
        out.append(
            {
                "start_step": int(t[0, 0]),
                "duration_s": float(dur),
                "f_min_mhz": float(f.min()),
                "f_max_mhz": float(f.max()),
                "amplitude_db": float(np.median(t[:, 2])),
                "wander_hz_s_p50": float(np.median(np.abs(np.diff(f))) * 1e6 / step_s) if len(f) > 1 else 0.0,
            }
        )
    return out
