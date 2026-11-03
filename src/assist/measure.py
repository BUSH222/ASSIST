"""IQ measurements for tests and validation: instantaneous frequency, tone frequency and C/N0.

All work in float64/complex128. Unwrapping a float32 phase loses resolution once the accumulated phase passes
~3e4 rad (a tone at +350 kHz gets there in 35k samples at 2.4 MS/s), so frequencies come from phase differences of
neighbouring samples instead.
"""

import numpy as np
import scipy.fft as sfft


def iq_from_u8(u8: np.ndarray) -> np.ndarray:
    """Interleaved uint8 I, Q -> complex128 in counts around 127.5."""
    x = np.asarray(u8, dtype=np.float64) - 127.5
    return x[0::2] + 1j * x[1::2]


def inst_freq(y: np.ndarray, fs: float) -> np.ndarray:
    """Instantaneous frequency (Hz) between neighbouring samples of a clean signal; length len(y) - 1."""
    z = np.asarray(y, dtype=np.complex128)
    return np.angle(z[1:] * np.conj(z[:-1])) * fs / (2 * np.pi)


def extract_band(y: np.ndarray, fs: float, f_centre: float, half_width: float) -> tuple[np.ndarray, float, float]:
    """Keep the FFT bins within f_centre +- half_width (baseband Hz) of the whole window, centred on 0 Hz.

    Returns (z, rate of z, the frequency that z's 0 Hz stands for: the bin nearest f_centre). z[m] is at time
    m / rate from the window start. The cut is a brick wall in the circular sense, so the ends of z ring; callers
    trim them.
    """
    n = len(y)
    spec = sfft.fft(np.asarray(y, dtype=np.complex128))
    qc = round(f_centre * n / fs)
    h = max(1, int(np.ceil(half_width * n / fs)))
    sel = np.arange(qc - h, qc + h + 1) % n  # 2h + 1 bins; the centre bin goes to index 0 below
    k = len(sel)
    z = sfft.ifft(np.roll(spec[sel], -h)) * (k / n)
    return np.asarray(z), k * fs / n, qc * fs / n


def tone_freq(
    y: np.ndarray, fs: float, f_guess: float, half_width: float = 250.0, trim: float = 0.1, deg: int = 3
) -> float:
    """Frequency (baseband Hz) of a tone near f_guess at the centre of the window (time len(y) / 2 / fs).

    The band f_guess +- half_width is cut out and decimated, its phase is unwrapped (the decimated signal is
    slow, so the phase stays small), and a polynomial of degree `deg` in time is fitted to the middle 1 - 2 * trim
    of the window. The slope at the centre is the frequency.
    """
    z, rate, f0 = extract_band(y, fs, f_guess, half_width)
    n = len(z)
    a, b = int(n * trim), int(n * (1 - trim))
    t = np.arange(n) / rate - len(y) / (2 * fs)
    ph = np.unwrap(np.angle(z))
    c = np.polynomial.polynomial.polyfit(t[a:b], ph[a:b], deg)
    return f0 + float(c[1]) / (2 * np.pi)


def cn0_dbhz(
    y: np.ndarray, fs: float, f_lo: float, f_hi: float, ring: tuple[float, float] = (300.0, 2000.0), guard: int = 4
) -> tuple[float, float, float]:
    """(C/N0 dB-Hz, carrier power, noise density) of a tone that stays within [f_lo, f_hi] (baseband Hz).

    Hann-windowed periodogram scaled to a density, so a tone's bins sum to its power and noise bins average to N0.
    N0 is the mean over bins ring[0]..ring[1] Hz outside the tone band on both sides; C is the band's power (guard
    bins added each side for the window's main lobe) minus N0 times the band width.
    """
    n = len(y)
    w = np.hanning(n + 2)[1:-1]
    p = np.abs(sfft.fft(np.asarray(y, dtype=np.complex128) * w)) ** 2 / (np.sum(w**2) * fs)
    df = fs / n
    f = sfft.fftfreq(n, 1 / fs)
    band = (f >= f_lo - guard * df) & (f <= f_hi + guard * df)
    noise = ((f >= f_lo - ring[1]) & (f <= f_lo - ring[0])) | ((f >= f_hi + ring[0]) & (f <= f_hi + ring[1]))
    n0 = float(np.mean(p[noise]))
    c = float(np.sum(p[band]) * df - n0 * np.count_nonzero(band) * df)
    return float(10 * np.log10(c / n0)), c, n0
