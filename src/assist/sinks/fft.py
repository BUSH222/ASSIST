"""FFT waterfall sink: the `fft` entry of a scenario's `sinks:`, a parametrised samples/waterfall.py.

    sinks:
      - fft: {width: 16384, row_seconds: 1, duration: 1h, min_db: 20, max_db: 50}

- **Rows:** a row is the mean of |FFT|^2 over `avg_count = max(1, round(row_seconds * fs / width))` consecutive,
  non-overlapping `width`-point FFTs (Nuttall window, fftshifted, float64 sums), so a row lasts
  `avg_count * width / fs` seconds, which is `row_seconds` to within one FFT. The image is `width` columns (low to
  high frequency) by `floor(duration * fs / row_samples)` rows (oldest at the top): its height is derived.
  `duration` is the sink's own limit; the stream of the run (`--duration`, else the scenario clock) is the other,
  and the shorter of the two applies, so the sink never runs open-ended.
- **Pixels:** `clip((10*log10(mean + 1e-12) - min_db) / (max_db - min_db) * 255, 0, 255)`, truncated to uint8, from
  IQ scaled as (u - 127.5) / 127.5. Power is scaled by 16384 / width, so a given dB means the same for any width
  (and equals the capture script's, which uses 16384).
- **Levels:** `min_db` and `max_db` together fix the scale. With neither, the script's rule applies: `min_db` is the
  mean over bins of the spectrum, in dB, of the first 5 s (as many whole rows as fit in them, at least one; the whole
  run if it is shorter) minus 5, and `max_db = min_db + 30`. Those rows are part of the image.
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, override

import numpy as np
from PIL import Image
from pydantic import Field, model_validator
from scipy.fft import fft
from scipy.signal.windows import nuttall

from assist.sinks.base import Sink, SinkContext, SinkParams, register_sink
from assist.sinks.wavfile import OutFile
from assist.timebase import parse_duration

if TYPE_CHECKING:
    from pathlib import Path

KEYS = frozenset({"width", "row_seconds", "duration", "min_db", "max_db"})
REF_WIDTH = 16384  # the capture script's FFT size; power is expressed as if it had been used
MIN_WIDTH, MAX_WIDTH = 64, 1 << 20
CAL_SECONDS = 5.0
CAL_FLOOR_DB = -5.0  # auto min_db = calibration mean - 5
AUTO_SPAN_DB = 30.0
_BATCH_SAMPLES = 1 << 21  # FFT blocks are transformed in groups of about this many samples


@dataclass(frozen=True)
class FftSpec:
    width: int = REF_WIDTH
    row_seconds: float = 1.0
    duration: float | None = None
    min_db: float | None = None
    max_db: float | None = None

    @staticmethod
    def from_dict(cfg: dict[str, Any] | None, duration: str | float | None = None) -> FftSpec:
        """Validate the sink's mapping; `duration` overrides the one in the mapping. Without either the spec has no
        duration yet: the sink takes the length of the stream it is given (`limited`)."""
        cfg = cfg or {}
        bad = sorted(set(cfg) - KEYS)
        if bad:
            raise ValueError(f"unknown fft sink key(s) {bad}; allowed: {sorted(KEYS)}")
        width = cfg.get("width", REF_WIDTH)
        if isinstance(width, bool) or not isinstance(width, int) or not MIN_WIDTH <= width <= MAX_WIDTH:
            raise ValueError(f"fft width must be an integer from {MIN_WIDTH} to {MAX_WIDTH}, got {width!r}")
        if width & (width - 1):
            raise ValueError(f"fft width must be a power of two, got {width}")
        row_seconds = parse_duration(cfg.get("row_seconds", 1.0))
        if not row_seconds > 0:
            raise ValueError("fft row_seconds must be positive")
        raw = duration if duration is not None else cfg.get("duration")
        dur = None if raw is None else parse_duration(raw)
        if dur is not None and not dur > 0:
            raise ValueError("fft duration must be positive")
        lo, hi = cfg.get("min_db"), cfg.get("max_db")
        if (lo is None) != (hi is None):
            raise ValueError("give both min_db and max_db, or neither (then the level is found from the data)")
        if lo is not None and hi is not None:
            if any(isinstance(v, bool) or not isinstance(v, int | float) for v in (lo, hi)):
                raise ValueError("fft min_db and max_db must be numbers")
            if not hi > lo:
                raise ValueError(f"fft max_db ({hi}) must be above min_db ({lo})")
            return FftSpec(width, row_seconds, dur, float(lo), float(hi))
        return FftSpec(width, row_seconds, dur)

    def limited(self, seconds: float) -> FftSpec:
        """The spec for a stream of `seconds`: its own duration, or the stream's when that is shorter (or unset)."""
        return replace(self, duration=seconds if self.duration is None else min(self.duration, seconds))

    def avg_count(self, fs: int) -> int:
        return max(1, round(self.row_seconds * fs / self.width))

    def row_samples(self, fs: int) -> int:
        return self.avg_count(fs) * self.width

    def height(self, fs: int) -> int:
        assert self.duration is not None
        return int(self.duration * fs) // self.row_samples(fs)


def waterfall_name(fc_hz: float, t_unix: float, tz: str) -> str:
    from assist.sinks.wavfile import sdrpp_name

    return sdrpp_name(fc_hz, t_unix, tz, "png").replace("baseband_", "waterfall_", 1)


class WaterfallSink:
    """Accumulates the uint8 IQ stream (interleaved I, Q; any piece sizes) into waterfall rows."""

    def __init__(self, spec: FftSpec, fs: int) -> None:
        self.spec, self.fs = spec, fs
        self.width = spec.width
        self.avg_count = spec.avg_count(fs)
        self.row_samples = spec.row_samples(fs)
        self.height = spec.height(fs)
        if self.height < 1:
            raise ValueError(
                f"duration {spec.duration} s is shorter than one row ({self.row_samples / fs:.3f} s): "
                "lengthen it or shorten row_seconds"
            )
        self.n_samples = self.height * self.row_samples
        self.image = np.zeros((self.height, self.width), dtype=np.uint8)
        self.rows = 0
        self.min_db, self.max_db = spec.min_db, spec.max_db
        self._cal_rows = (
            0 if spec.min_db is not None else min(self.height, max(1, int(CAL_SECONDS * fs) // self.row_samples))
        )
        self._held: list[np.ndarray] = []  # mean power of the rows seen before the level is known
        self._buf = np.empty(0, dtype=np.uint8)
        self._window = nuttall(self.width)
        self._scale = REF_WIDTH / self.width

    def write(self, u8: np.ndarray) -> None:
        piece = np.asarray(u8, dtype=np.uint8)
        self._buf = piece.copy() if self._buf.size == 0 else np.concatenate((self._buf, piece))
        step = 2 * self.row_samples
        done = 0
        while len(self._buf) - done >= step and self.rows + len(self._held) < self.height:
            self._row(self._buf[done : done + step])
            done += step
        self._buf = self._buf[done:].copy()

    def _mean_power(self, u8: np.ndarray) -> np.ndarray:
        """Mean |FFT|^2 (float64, fftshifted, scaled to the reference width) over one row's blocks."""
        acc = np.zeros(self.width, dtype=np.float64)
        group = max(1, _BATCH_SAMPLES // self.width)
        per_block = 2 * self.width
        for b in range(0, self.avg_count, group):
            k = min(group, self.avg_count - b)
            f = u8[b * per_block : (b + k) * per_block].astype(np.float32)
            iq = (f[0::2] - 127.5) / 127.5 + 1j * (f[1::2] - 127.5) / 127.5
            spec = np.fft.fftshift(fft(iq.reshape(k, self.width) * self._window, axis=-1, workers=-1), axes=-1)
            acc += (spec.real**2 + spec.imag**2).sum(axis=0, dtype=np.float64)
        return acc * (self._scale / self.avg_count)

    def _row(self, u8: np.ndarray) -> None:
        p = self._mean_power(u8)
        if self.min_db is None:
            self._held.append(p)
            if len(self._held) < self._cal_rows:
                return
            mean = np.mean(self._held, axis=0)
            self.min_db = float((10 * np.log10(mean + 1e-12)).mean() + CAL_FLOOR_DB)
            self.max_db = self.min_db + AUTO_SPAN_DB
            held, self._held = self._held, []
            for q in held:
                self._pixels(q)
        else:
            self._pixels(p)

    def _pixels(self, mean_power: np.ndarray) -> None:
        assert self.min_db is not None
        assert self.max_db is not None
        db = 10 * np.log10(mean_power + 1e-12)
        self.image[self.rows] = np.clip((db - self.min_db) / (self.max_db - self.min_db) * 255, 0, 255).astype(
            np.uint8
        )
        self.rows += 1

    def close(self) -> None:
        """Complete when every planned sample arrived; raises otherwise."""
        if self.rows != self.height:
            raise ValueError(f"made {self.rows} rows, planned {self.height}")

    def save(self, path: Path) -> None:
        """Write the PNG of the finished rows (an interrupted run keeps them; rows held for the level are lost)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".png.part")
        Image.fromarray(self.image[: self.rows]).save(tmp, format="PNG")
        tmp.replace(path)

    def info(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "rows_written": self.rows,
            "row_seconds_requested": self.spec.row_seconds,
            "row_seconds_actual": self.row_samples / self.fs,
            "avg_count": self.avg_count,
            "bin_hz": self.fs / self.width,
            "min_db": self.min_db,
            "max_db": self.max_db,
            "levels": "given" if self.spec.min_db is not None else "calibrated",
        }


class FftParams(SinkParams):
    width: int = Field(REF_WIDTH, description="FFT size and image width: a power of two from 64 to 1048576.")
    row_seconds: float = Field(1.0, description="Seconds of stream per image row (strings like 250ms are accepted).")
    duration: float | None = Field(
        None, description="Plot at most this much stream, e.g. 10min (default: the whole stream of the run)."
    )
    min_db: float | None = Field(None, description="Level of black; with max_db, or neither (found from the data).")
    max_db: float | None = Field(None, description="Level of white.")

    @model_validator(mode="before")
    @classmethod
    def _check(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return v
        spec = FftSpec.from_dict(v)  # one set of rules and messages for the mapping
        return {
            "width": spec.width,
            "row_seconds": spec.row_seconds,
            "duration": spec.duration,
            "min_db": spec.min_db,
            "max_db": spec.max_db,
        }


@register_sink("fft")
class FftSink(Sink):
    """The waterfall PNG of the stream (see the module text)."""

    Params = FftParams

    def __init__(self, params: SinkParams, ctx: SinkContext) -> None:
        super().__init__(params, ctx)
        assert isinstance(params, FftParams) and ctx.n is not None
        spec = FftSpec(params.width, params.row_seconds, params.duration, params.min_db, params.max_db)
        self.w = WaterfallSink(spec.limited(ctx.n / ctx.fs), ctx.fs)
        self.png = ctx.out / waterfall_name(ctx.fc_hz, ctx.n0 / ctx.fs, ctx.tz)
        self._saved = True  # until the run ends, the planned file is what `files` reports

    @override
    def planned(self) -> int:
        return self.w.n_samples

    @override
    def files(self) -> list[OutFile]:
        return [OutFile(self.png, 0, self.w.n_samples, self.ctx.n0)] if self._saved else []

    @override
    def write(self, u8: np.ndarray) -> None:
        self.w.write(u8)

    def _save(self) -> None:
        self._saved = self.w.rows > 0
        if self._saved:
            self.w.save(self.png)

    @override
    def close(self) -> None:
        try:
            self.w.close()
        finally:
            self._save()

    @override
    def abort(self) -> None:
        self._save()

    @override
    def info(self) -> dict[str, Any]:
        return self.w.info()

    @override
    def summary(self) -> str:
        w = self.w
        if w.min_db is None or w.max_db is None:
            return f"fft: {w.rows}/{w.height} rows (level not yet calibrated)"
        return f"fft: {w.rows}/{w.height} rows, {w.width} px wide, levels {w.min_db:.1f}..{w.max_db:.1f} dB"
