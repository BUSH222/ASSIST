"""The station's waterfall maths, re-implemented exactly from the capture script (samples/waterfall.py).

The capture script reads an rtl_tcp stream and, per row:
- converts bytes to complex64 as (u - 127.5) / 127.5 (I and Q);
- takes `ROW_FFTS` consecutive, non-overlapping FFT_SIZE-point FFTs with a Nuttall window (scipy.fft, fftshifted);
- averages |X|^2 in float64 and maps 10*log10(mean + 1e-12) to a byte: clip((dB - min_db) / 30 * 255, 0, 255),
  truncated (astype(uint8), not rounded);
- `min_db` comes once per file from the first CAL_FFTS FFTs: mean over bins of their averaged spectrum in dB, minus 5.

So pixel p means dB relative to the calibration mean in [p, p+1) * 30/255 - 5; 0 and 255 are saturated.
Row i covers samples [CAL_SAMPLES + i*ROW_SAMPLES, CAL_SAMPLES + (i+1)*ROW_SAMPLES) of the stream.
"""

from functools import cache
from typing import TYPE_CHECKING

import numpy as np
from scipy.fft import fft
from scipy.signal.windows import nuttall

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

FS = 2_400_000
FFT_SIZE = 16384
SPAN_DB = 30.0
FLOOR_OFFSET_DB = -5.0  # min_db = calibration mean - 5
CAL_FFTS = max(1, int(5.0 * FS / FFT_SIZE))  # 732
ROW_FFTS = max(1, round(1.0 * FS / FFT_SIZE))  # 146
CAL_SAMPLES = CAL_FFTS * FFT_SIZE  # 11 993 088
ROW_SAMPLES = ROW_FFTS * FFT_SIZE  # 2 392 064
ROW_SECONDS = ROW_SAMPLES / FS  # 0.99669 s
DB_PER_LSB = SPAN_DB / 255.0


@cache
def window() -> np.ndarray:
    """Nuttall window of FFT_SIZE points (float64, read-only)."""
    w = nuttall(FFT_SIZE)
    w.flags.writeable = False
    return w


def iq_from_u8(u8: np.ndarray) -> np.ndarray:
    """Interleaved I,Q bytes -> complex64, scaled exactly like the capture script."""
    f = np.asarray(u8, dtype=np.uint8).astype(np.float32)
    return (f[0::2] - 127.5) / 127.5 + 1j * (f[1::2] - 127.5) / 127.5


def power_spectra(iq: np.ndarray) -> np.ndarray:
    """|FFT|^2 of consecutive FFT_SIZE blocks, fftshifted, float64. iq: (k*FFT_SIZE,) -> (k, FFT_SIZE)."""
    blocks = iq.reshape(-1, FFT_SIZE)
    spec = np.fft.fftshift(fft(blocks * window(), axis=-1, workers=-1), axes=-1)
    return spec.real**2 + spec.imag**2


def mean_power(iq: np.ndarray) -> np.ndarray:
    """Average power spectrum of consecutive FFT blocks, summed in order like the capture script."""
    p = power_spectra(iq)
    acc = np.zeros(FFT_SIZE, dtype=np.float64)
    for row in p:
        acc += row
    return acc / p.shape[0]


def power_db(mean_pow: np.ndarray) -> np.ndarray:
    return 10 * np.log10(mean_pow + 1e-12)


def calibration_min_db(cal_iq: np.ndarray) -> float:
    """min_db from the calibration samples (CAL_FFTS blocks)."""
    return float(power_db(mean_power(cal_iq)).mean() + FLOOR_OFFSET_DB)


def to_pixels(row_db: np.ndarray, min_db: float) -> np.ndarray:
    max_db = min_db + SPAN_DB
    return np.clip((row_db - min_db) / (max_db - min_db) * 255, 0, 255).astype(np.uint8)


def pixel_to_db(p: np.ndarray, centre: bool = False) -> np.ndarray:
    """Pixel -> dB relative to the calibration mean. centre=True returns the middle of the truncation interval."""
    pf = np.asarray(p, dtype=np.float32)
    return (pf + (0.5 if centre else 0.0)) * np.float32(DB_PER_LSB) + np.float32(FLOOR_OFFSET_DB)


def bin_freqs(fc_hz: float, fs: float = FS) -> np.ndarray:
    """Centre frequency (Hz) of each fftshifted bin."""
    return fc_hz + (np.arange(FFT_SIZE) - FFT_SIZE // 2) * (fs / FFT_SIZE)


def n_rows(n_samples: int) -> int:
    """Number of complete rows the capture script gets from a stream of n_samples."""
    return max(0, (n_samples - CAL_SAMPLES) // ROW_SAMPLES)


def row_start_sample(i: int | np.ndarray) -> int | np.ndarray:
    return CAL_SAMPLES + np.asarray(i) * ROW_SAMPLES


def rows(read_iq: Callable[[int, int], np.ndarray], n_samples: int) -> Iterator[tuple[int, float, np.ndarray]]:
    """Waterfall of a sample stream: yields (row index, min_db, row dB as float64) for every complete row.

    read_iq(n0, n) returns complex64 samples [n0, n0+n) (e.g. Recording.iq). Pixels: to_pixels(row_db, min_db).
    """
    min_db = calibration_min_db(read_iq(0, CAL_SAMPLES))
    for i in range(n_rows(n_samples)):
        yield i, min_db, power_db(mean_power(read_iq(int(row_start_sample(i)), ROW_SAMPLES)))
