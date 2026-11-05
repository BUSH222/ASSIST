"""Expensive per-file intermediates, optionally cached on disk (tool outputs, e.g. out/profile/cache/).

- WAV waterfall rows: the capture-script maths applied to the recording (float32 dB, one row per 0.997 s).
- Day summaries of a .raw waterfall: per-minute mean (dB, centred pixels), per-minute max (pixels), and a per-row
  band level (trimmed mean dB over 136.1-138.15 MHz).

Cache files are keyed by input name and size; delete the directory to force a rebuild.
"""

from typing import TYPE_CHECKING

import numpy as np

from assist import waterfall as wf

if TYPE_CHECKING:
    from pathlib import Path

    from assist.profile.rawio import RawWaterfall
    from assist.profile.wavio import Recording

ROWS_PER_MINUTE = 60
BAND_MHZ = (136.1, 138.15)


def _key(path: Path) -> str:
    return f"{path.stem}_{path.stat().st_size}"


def wav_rows(rec: Recording, cache_dir: Path | None = None) -> tuple[np.ndarray, float]:
    """(rows dB float32 (n_rows, FFT_SIZE), min_db) of the recording, capture-script maths."""
    if cache_dir is not None:
        f = cache_dir / f"wavrows_{_key(rec.path)}.npz"
        if f.exists():
            z = np.load(f)
            return z["rows"], float(z["min_db"])
    n = wf.n_rows(rec.n_samples)
    rows = np.empty((n, wf.FFT_SIZE), dtype=np.float32)
    min_db = 0.0
    for i, m, row in wf.rows(rec.iq, rec.n_samples):
        rows[i] = row
        min_db = m
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache_dir / f"wavrows_{_key(rec.path)}.npz", rows=rows, min_db=min_db)
    return rows, min_db


def centred_lut() -> np.ndarray:
    """Pixel -> dB (middle of the truncation interval), 256 entries."""
    return wf.pixel_to_db(np.arange(256), centre=True)


def day_summary(raw: RawWaterfall, cache_dir: Path | None = None) -> dict[str, np.ndarray]:
    """Per-minute mean dB, per-minute max pixel, and per-row band level of a .raw waterfall."""
    if cache_dir is not None:
        f = cache_dir / f"day_{_key(raw.path)}.npz"
        if f.exists():
            return dict(np.load(f))
    lut = centred_lut()
    fmhz = raw.freqs() / 1e6
    band = (fmhz > BAND_MHZ[0]) & (fmhz < BAND_MHZ[1])
    n_min = raw.n_rows // ROWS_PER_MINUTE
    mean_db = np.empty((n_min, wf.FFT_SIZE), np.float32)
    max_px = np.empty((n_min, wf.FFT_SIZE), np.uint8)
    level = np.empty(raw.n_rows, np.float32)
    step = 60 * ROWS_PER_MINUTE
    for r0 in range(0, raw.n_rows, step):
        px = np.asarray(raw.pixels[r0 : r0 + step])
        db = lut[px]
        q = np.sort(db[:, band], axis=1)
        k = q.shape[1]
        level[r0 : r0 + len(px)] = q[:, k // 10 : 9 * k // 10].mean(axis=1)
        m = len(px) // ROWS_PER_MINUTE
        i0 = r0 // ROWS_PER_MINUTE
        mean_db[i0 : i0 + m] = db[: m * ROWS_PER_MINUTE].reshape(m, ROWS_PER_MINUTE, -1).mean(axis=1)
        max_px[i0 : i0 + m] = px[: m * ROWS_PER_MINUTE].reshape(m, ROWS_PER_MINUTE, -1).max(axis=1)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_dir / f"day_{_key(raw.path)}.npz", minute_mean_db=mean_db, minute_max_px=max_px, row_level_db=level
        )
    return {"minute_mean_db": mean_db, "minute_max_px": max_px, "row_level_db": level}
