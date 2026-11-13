"""Event list and label sheet.

The build writes every detected event of the waterfall days to <work>/events.csv. `profile labels` draws a
stratified sample (about 200 events), renders a crop for each (+-50 kHz, 60 s before and after, from the .raw file)
and writes events.csv with an empty `label` column to fill from the fixed vocabulary.
"""

import csv
from typing import TYPE_CHECKING, Any

import numpy as np

from assist import waterfall as wf
from assist.profile.cache import centred_lut

if TYPE_CHECKING:
    from pathlib import Path

    from assist.profile.rawio import RawWaterfall

VOCABULARY = (
    "vdl",
    "am",
    "meteor",
    "orbcomm_stx",
    "orbcomm_gw",
    "lora",
    "carrier_138",
    "mirror",
    "usb_hump",
    "birdie",
    "unknown_sat",
    "unknown",
    "noise",
)
FIELDS = ("id", "file", "row_start", "row_end", "utc_start", "utc_end", "f_lo_hz", "f_hi_hz", "auto_label", "peak_db")


def satband_events(
    minute_max_px: np.ndarray, baseline_db: np.ndarray, f_hz: np.ndarray, known_mask: np.ndarray, on_db: float = 8.0
) -> list[tuple[int, int, int, int, float]]:
    """Unexplained activity in 137.0-138.0 MHz: (minute0, minute1, bin_lo, bin_hi, peak dB) clusters of per-minute
    max pixels > on_db over the hour's baseline, outside `known_mask` (minutes x bins: known satellites, VDL, birdies).
    """
    from scipy.ndimage import find_objects, label

    lut = centred_lut()
    band = (f_hz >= 137.0e6) & (f_hz <= 138.0e6)
    exc = lut[minute_max_px] - baseline_db[np.minimum(np.arange(len(minute_max_px)) // 60, len(baseline_db) - 1)]
    hot = (exc > on_db) & band[None, :] & ~known_mask
    lab, _n = label(hot, structure=np.ones((3, 3)))
    out = []
    for i, sl in enumerate(find_objects(lab), start=1):
        if sl is None:
            continue
        m0, m1 = sl[0].start, sl[0].stop
        b0, b1 = sl[1].start, sl[1].stop
        if (lab[sl] == i).sum() < 2:
            continue
        out.append((int(m0), int(m1), int(b0), int(b1), float(exc[sl][lab[sl] == i].max())))
    return out


def write_events(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in FIELDS})


def read_events(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def stratified_sample(events: list[dict[str, Any]], total: int = 200, seed: int = 0) -> list[dict[str, Any]]:
    """Up to `total` events, at least min(total/classes, n_class) per auto-label class, the rest proportional."""
    rng = np.random.default_rng(seed)  # picks events for the label sheet; never feeds a render
    by: dict[str, list[dict[str, Any]]] = {}
    for e in events:
        by.setdefault(e["auto_label"], []).append(e)
    per = max(1, total // max(1, len(by)))
    out: list[dict[str, Any]] = []
    for _k, v in sorted(by.items()):
        idx = rng.permutation(len(v))[:per]
        out += [v[i] for i in sorted(idx)]
    return out[:total]


def crop_png(raw: RawWaterfall, row0: int, row1: int, f_lo: float, f_hi: float, path: Path) -> None:
    """PNG crop: +-50 kHz around the event and 60 s before/after, capture-script pixel values."""
    from PIL import Image

    f = raw.freqs()
    fc = (f_lo + f_hi) / 2
    half = max(50e3, (f_hi - f_lo) / 2 + 10e3)
    k0, k1 = np.searchsorted(f, [fc - half, fc + half])
    pad = int(60 / wf.ROW_SECONDS)
    r0, r1 = max(0, row0 - pad), min(raw.n_rows, row1 + pad)
    img = np.asarray(raw.pixels[r0:r1, k0:k1])
    # mark the event's time span on the left edge
    img = np.concatenate([np.zeros((len(img), 4), np.uint8), img], axis=1)
    img[row0 - r0 : row1 - r0, :3] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


__all__ = ["VOCABULARY", "centred_lut"]
