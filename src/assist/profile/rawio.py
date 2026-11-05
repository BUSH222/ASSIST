"""Waterfall .raw loader (plan §9, P4/P5).

A .raw file is the capture script's byte rows: n_rows x FFT_SIZE uint8, no header. Its name holds the Unix time
taken just before connecting (`waterfall_<int(time.time())>.raw`), so row times carry up to ~1 s of offset from
truncation and connection latency, plus 6 samples from the unread 12-byte rtl_tcp header. Rows are ~1 s, so this
is below the time resolution.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from assist.waterfall import CAL_SAMPLES, FFT_SIZE, FS, ROW_SAMPLES, bin_freqs, pixel_to_db

_RAW_NAME = re.compile(r"waterfall_(?P<t>\d+)\.raw$")


@dataclass
class RawWaterfall:
    path: Path
    t_file: int  # Unix seconds from the file name
    pixels: np.memmap  # (n_rows, FFT_SIZE) uint8, read-only
    fs: float = FS

    @property
    def n_rows(self) -> int:
        return int(self.pixels.shape[0])

    def row_time(self, i: int | np.ndarray) -> float | np.ndarray:
        """Unix time of the middle of row i: t_file + CAL/fs + (i + 0.5) * ROW/fs (P5)."""
        return self.t_file + CAL_SAMPLES / self.fs + (np.asarray(i, dtype=np.float64) + 0.5) * ROW_SAMPLES / self.fs

    def row_at(self, t_unix: float) -> int:
        """Index of the row containing unix time t (may be out of range)."""
        return int(np.floor((t_unix - self.t_file - CAL_SAMPLES / self.fs) * self.fs / ROW_SAMPLES))

    def db(self, i0: int, i1: int, centre: bool = False) -> np.ndarray:
        """Rows [i0, i1) in dB relative to the file's calibration mean (P4), float32."""
        return pixel_to_db(self.pixels[i0:i1], centre=centre)

    def freqs(self, fc_hz: float = 137.0e6) -> np.ndarray:
        return bin_freqs(fc_hz, self.fs)


def open_raw(path: str | Path) -> RawWaterfall:
    path = Path(path)
    m = _RAW_NAME.search(path.name)
    if not m:
        raise ValueError(f"not a waterfall_<unix>.raw file: {path.name}")
    size = path.stat().st_size
    if size % FFT_SIZE:
        raise ValueError(f"{path.name}: size {size} is not a multiple of {FFT_SIZE}")
    pixels = np.memmap(path, dtype=np.uint8, mode="r", shape=(size // FFT_SIZE, FFT_SIZE))
    return RawWaterfall(path, int(m.group("t")), pixels)
