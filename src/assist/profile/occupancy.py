"""P7: channel occupancy.

Waterfall days (1 s rows): per row, excess over an hourly baseline spectrum (median of the hour's per-minute means);
a channel's excess is the max over its bins. On/off with hysteresis (on above `on_db`, off below `off_db`) gives
events -> hourly rates, durations (row resolution, ~1 s) and peak levels.
Recording (fine time): 1024-point FFT frames (0.43 ms) summed over each channel give burst durations, gaps and
levels that 1 s rows smear out.

Channels: the VDL2 25 kHz grid 136.700-136.975 MHz (+-10 kHz), the airband 8.33 kHz grid 135.800-136.700 MHz
(+-3 kHz), and 138.0-138.2 MHz scanned as one band (carriers at any frequency; the peak bin gives the frequency).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from assist.profile.cache import ROWS_PER_MINUTE, centred_lut
from assist.profile.dsp import hysteresis, runs

if TYPE_CHECKING:
    from assist.profile.rawio import RawWaterfall
    from assist.profile.wavio import Recording


@dataclass(frozen=True)
class Channel:
    name: str
    kind: str  # vdl | am | band138
    f_lo_hz: float
    f_hi_hz: float

    @property
    def centre_hz(self) -> float:
        return (self.f_lo_hz + self.f_hi_hz) / 2


def default_channels() -> list[Channel]:
    chans = [Channel(f"vdl_{f / 1e6:.3f}", "vdl", f - 10e3, f + 10e3) for f in np.arange(136.700e6, 136.976e6, 25e3)]
    am_grid = 135.800e6 + np.arange(0, 108) * 25e3 / 3
    chans += [Channel(f"am_{f / 1e6:.4f}", "am", f - 3e3, f + 3e3) for f in am_grid if f < 136.700e6 - 3e3]
    chans.append(Channel("band_138.0-138.2", "band138", 138.0e6, 138.2e6))
    return chans


def channel_bins(chans: list[Channel], f_hz: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero((f_hz >= c.f_lo_hz) & (f_hz <= c.f_hi_hz)) for c in chans]


def hourly_baseline(minute_mean_db: np.ndarray) -> np.ndarray:
    """(n_hours, FFT_SIZE): per-bin median of each hour's per-minute means."""
    n_h = int(np.ceil(len(minute_mean_db) / 60))
    return np.stack([np.median(minute_mean_db[h * 60 : (h + 1) * 60], axis=0) for h in range(n_h)])


def day_channel_excess(
    raw: RawWaterfall, baseline: np.ndarray, chans: list[Channel], f_hz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(excess dB (n_rows, n_chan) as float16, peak bin per row for each channel (n_rows, n_chan) int16)."""
    lut = centred_lut()
    bins = channel_bins(chans, f_hz)
    exc = np.empty((raw.n_rows, len(chans)), np.float16)
    peak = np.empty((raw.n_rows, len(chans)), np.int16)
    step = 3600
    for r0 in range(0, raw.n_rows, step):
        db = lut[np.asarray(raw.pixels[r0 : r0 + step])]
        h = np.minimum((r0 + np.arange(len(db))) // (60 * ROWS_PER_MINUTE), len(baseline) - 1)
        x = db - baseline[h]
        for j, b in enumerate(bins):
            sub = x[:, b]
            k = np.argmax(sub, axis=1)
            exc[r0 : r0 + len(db), j] = sub[np.arange(len(sub)), k]
            peak[r0 : r0 + len(db), j] = b[k]
    return exc, peak


def events_from_series(excess_db: np.ndarray, on_db: float, off_db: float) -> np.ndarray:
    """[start, end) indices of events (hysteresis)."""
    return runs(hysteresis(excess_db.astype(np.float32), on_db, off_db))


def lognormal_fit(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[x > 0]
    if len(x) < 3:
        return {"n": len(x)}
    lx = np.log(x)
    return {
        "n": len(x),
        "mu": float(lx.mean()),
        "sigma": float(lx.std(ddof=1)),
        "median": float(np.exp(np.median(lx))),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
    }


def stft_channel_power(
    rec: Recording, chans: list[Channel], n0: int, n1: int, nfft: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
    """(frame start sample (n_frames,), channel power (n_frames, n_chan)) from non-overlapping Hann FFT frames.

    VDL/AM channels sum their bins; the 138 MHz band takes its strongest bin (a narrow carrier would be diluted by
    ~19 dB in a 200 kHz sum).
    """
    f = (np.arange(nfft) - nfft // 2) * rec.fs / nfft + 137.0e6
    bins = [np.flatnonzero((f >= c.f_lo_hz) & (f <= c.f_hi_hz)) for c in chans]
    win = np.hanning(nfft).astype(np.float32)
    starts, powers = [], []
    chunk = nfft * 2048
    for a in range(n0, n1 - nfft + 1, chunk):
        m = min(chunk, (n1 - a) // nfft * nfft)
        x = rec.iq(a, m).reshape(-1, nfft) * win
        p = np.abs(np.fft.fftshift(np.fft.fft(x, axis=1), axes=1)) ** 2
        cols = []
        for c, b in zip(chans, bins, strict=True):
            if not len(b):
                cols.append(np.zeros(len(p)))
            else:
                cols.append(p[:, b].max(axis=1) if c.kind == "band138" else p[:, b].sum(axis=1))
        powers.append(np.stack(cols, axis=1))
        starts.append(a + np.arange(len(p)) * nfft)
    return np.concatenate(starts), np.concatenate(powers).astype(np.float32)


def bursts(
    power: np.ndarray, frame_s: float, on_db: float = 10.0, off_db: float = 5.0, smooth: int = 3
) -> dict[str, Any]:
    """Burst statistics of one channel's fine-time power series (noise = 20th percentile)."""
    noise = float(np.percentile(power, 20))
    x = np.convolve(power, np.ones(smooth) / smooth, mode="same")
    exc = 10 * np.log10(np.maximum(x, 1e-30) / noise)
    ev = runs(hysteresis(exc, on_db, off_db))
    if len(ev) == 0:
        return {"events": 0}
    dur = (ev[:, 1] - ev[:, 0]) * frame_s
    gaps = (ev[1:, 0] - ev[:-1, 1]) * frame_s
    peak = np.array([exc[s:e].max() for s, e in ev])
    return {
        "events": len(ev),
        "duty": float((ev[:, 1] - ev[:, 0]).sum() / len(power)),
        "duration_s": lognormal_fit(dur),
        "gap_s": {"mean": float(gaps.mean()) if len(gaps) else None, **lognormal_fit(gaps)},
        "peak_excess_db_p10_p50_p90": [float(v) for v in np.percentile(peak, [10, 50, 90])],
    }
