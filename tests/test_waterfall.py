"""assist.waterfall reproduces the capture script (samples/waterfall.py) byte for byte.

`_script_*` below are verbatim copies of the capture script's maths (only the socket replaced by a byte buffer).
"""

import numpy as np
import pytest
from scipy.fft import fft
from scipy.signal.windows import nuttall

from assist import waterfall as wf

SAMPLE_RATE = 2_400_000
FFT_SIZE = 16384
FFT_MAX_DIFF_DB = 30
CALIBRATION_SECONDS = 5.0
ROW_SECONDS = 1.0


class _Stream:
    def __init__(self, data: bytes) -> None:
        self.data, self.pos = data, 0

    def read_iq(self, n_samples: int) -> np.ndarray:
        raw = self.data[self.pos : self.pos + 2 * n_samples]
        self.pos += 2 * n_samples
        u8 = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (u8[0::2] - 127.5) / 127.5 + 1j * (u8[1::2] - 127.5) / 127.5


def _script_power_spectrum(iq: np.ndarray, window: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fftshift(fft(iq * window, workers=-1))
    return spectrum.real**2 + spectrum.imag**2


def _script_rows(data: bytes, n_rows: int) -> tuple[float, np.ndarray]:
    sock = _Stream(data)
    window = nuttall(FFT_SIZE)
    avg_count = max(1, round(ROW_SECONDS * SAMPLE_RATE / FFT_SIZE))
    n_blocks = max(1, int(CALIBRATION_SECONDS * SAMPLE_RATE / FFT_SIZE))
    power_sum = np.zeros(FFT_SIZE, dtype=np.float64)
    for _ in range(n_blocks):
        power_sum += _script_power_spectrum(sock.read_iq(FFT_SIZE), window)
    noise_db = 10 * np.log10(power_sum / n_blocks + 1e-12)
    min_db = noise_db.mean() - 5
    max_db = min_db + FFT_MAX_DIFF_DB
    rows = np.zeros((n_rows, FFT_SIZE), dtype=np.uint8)
    for r in range(n_rows):
        power_sum = np.zeros(FFT_SIZE, dtype=np.float64)
        for _ in range(avg_count):
            power_sum += _script_power_spectrum(sock.read_iq(FFT_SIZE), window)
        power_db = 10 * np.log10(power_sum / avg_count + 1e-12)
        rows[r] = np.clip((power_db - min_db) / (max_db - min_db) * 255, 0, 255).astype(np.uint8)
    return float(min_db), rows


def test_constants_match_plan() -> None:
    assert (wf.CAL_FFTS, wf.ROW_FFTS) == (732, 146)
    assert (wf.CAL_SAMPLES, wf.ROW_SAMPLES) == (11_993_088, 2_392_064)
    assert pytest.approx(0.99669, abs=1e-5) == wf.ROW_SECONDS


def test_matches_capture_script_byte_for_byte() -> None:
    n_rows = 2
    n = wf.CAL_SAMPLES + n_rows * wf.ROW_SAMPLES
    rng = np.random.default_rng(7)
    t = np.arange(n)
    # Noise + a tone that switches on during row 1 + a strong tone that saturates.
    i = 127.5 + 12 * rng.standard_normal(n) + 30 * np.cos(2 * np.pi * 0.1 * t) * (t > wf.CAL_SAMPLES + wf.ROW_SAMPLES)
    q = 127.5 + 12 * rng.standard_normal(n) + 30 * np.sin(2 * np.pi * 0.1 * t) * (t > wf.CAL_SAMPLES + wf.ROW_SAMPLES)
    i += 60 * np.cos(2 * np.pi * -0.3 * t)
    q += 60 * np.sin(2 * np.pi * -0.3 * t)
    u8 = np.empty(2 * n, dtype=np.uint8)
    u8[0::2] = np.clip(np.round(i), 0, 255)
    u8[1::2] = np.clip(np.round(q), 0, 255)

    want_min_db, want_rows = _script_rows(u8.tobytes(), n_rows)

    def read(n0: int, k: int) -> np.ndarray:
        return wf.iq_from_u8(u8[2 * n0 : 2 * (n0 + k)])

    got = list(wf.rows(read, n))
    assert [g[0] for g in got] == list(range(n_rows))
    assert got[0][1] == want_min_db
    got_rows = np.stack([wf.to_pixels(g[2], g[1]) for g in got])
    np.testing.assert_array_equal(got_rows, want_rows)
    assert want_rows.max() == 255  # the saturation path is exercised


def test_pixel_to_db() -> None:
    p = np.array([0, 255, 85])
    np.testing.assert_allclose(wf.pixel_to_db(p), [-5.0, 25.0, 5.0], atol=1e-5)
    np.testing.assert_allclose(wf.pixel_to_db(p, centre=True) - wf.pixel_to_db(p), 15 / 255, atol=1e-5)  # float32


def test_bin_freqs_and_rows() -> None:
    f = wf.bin_freqs(137.0e6)
    assert f[wf.FFT_SIZE // 2] == 137.0e6
    assert f[1] - f[0] == pytest.approx(2.4e6 / 16384)
    assert f[0] == pytest.approx(135.8e6)
    assert wf.n_rows(wf.CAL_SAMPLES + 3 * wf.ROW_SAMPLES - 1) == 2
    assert wf.n_rows(100) == 0
