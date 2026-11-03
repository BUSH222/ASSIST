"""cu8 file sinks (plan §7.2): WAV like the station's SDR++ recordings, single-file RF64, or headerless .cu8.

- **WAV:** PCM, 2 channels, 8-bit unsigned, fs. The header is the canonical 44 bytes (RIFF, fmt, data, nothing
  else): SDR++'s file source reads a fixed header layout. Files are split every 10 min (2.88 GB at 2.4 MS/s, under
  the 4 GiB RIFF limit).
- **RF64:** one file of any length (RF64, ds64, fmt, data). SDR++ 1.2.1's WAV source does not read RF64.
- **cu8:** one headerless file of interleaved I, Q bytes, as rtl_sdr writes.

Names follow SDR++ (`baseband_137000000Hz_HH-MM-SS_DD-MM-YYYY.wav`) with the file's first sample time in station
local time, truncated to the second. Headers are written with the final sizes up front, so an interrupted render
leaves files whose sizes are too large; readers that tolerate wrong sizes (assist's, SDR++) read to the end.
"""

import datetime as dt
import math
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

from assist.timebase import format_iso, parse_tz

if TYPE_CHECKING:
    from pathlib import Path
    from typing import BinaryIO

    import numpy as np

SPLIT_S = 600
FORMATS = ("wav", "rf64", "cu8")
_U32_MAX = 0xFFFFFFFF


def sdrpp_name(fc_hz: float, t_unix: float, tz: str, ext: str) -> str:
    local = dt.datetime.fromtimestamp(math.floor(t_unix), parse_tz(tz))
    return f"baseband_{round(fc_hz)}Hz_{local:%H-%M-%S}_{local:%d-%m-%Y}.{ext}"


def _fmt_chunk(fs: int) -> bytes:
    return b"fmt " + struct.pack("<IHHIIHH", 16, 1, 2, fs, 2 * fs, 2, 8)


def wav_header(n_samples: int, fs: int) -> bytes:
    data = 2 * n_samples
    if 36 + data > _U32_MAX:
        raise ValueError("a RIFF WAV cannot hold more than 4 GiB; split it or use RF64")
    return b"RIFF" + struct.pack("<I", 36 + data) + b"WAVE" + _fmt_chunk(fs) + b"data" + struct.pack("<I", data)


def rf64_header(n_samples: int, fs: int) -> bytes:
    data = 2 * n_samples
    ds64 = b"ds64" + struct.pack("<IQQQI", 28, 4 + 36 + 24 + 8 + data, data, n_samples, 0)
    return (
        b"RF64" + struct.pack("<I", _U32_MAX) + b"WAVE" + ds64 + _fmt_chunk(fs) + b"data" + struct.pack("<I", _U32_MAX)
    )


@dataclass(frozen=True)
class OutFile:
    path: Path
    first: int  # first sample, counted from the start of the render
    n_samples: int
    start_sample: int  # absolute sample index (Unix time * fs) of the first sample

    def info(self, fs: int) -> dict[str, object]:
        return {
            "name": self.path.name,
            "first_sample": self.first,
            "n_samples": self.n_samples,
            "start_sample_abs": self.start_sample,
            "start_utc": format_iso(self.start_sample / fs, 6),
            "duration_s": self.n_samples / fs,
        }


def plan_files(
    out_dir: Path, fmt: str, fc_hz: float, fs: int, tz: str, n0: int, n: int, split_s: float = SPLIT_S
) -> list[OutFile]:
    """The output files of a render of absolute samples [n0, n0 + n)."""
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}, got {fmt!r}")
    if n <= 0:
        raise ValueError("nothing to render")
    split = round(split_s * fs) if fmt == "wav" else n
    ext = "cu8" if fmt == "cu8" else "wav"
    files = []
    for first in range(0, n, split):
        start = n0 + first
        files.append(OutFile(out_dir / sdrpp_name(fc_hz, start / fs, tz, ext), first, min(split, n - first), start))
    names = [f.path.name for f in files]
    if len(set(names)) != len(names):
        raise ValueError("output files would share a name (split shorter than 1 s?)")
    return files


class FileSink:
    """Writes a uint8 stream (interleaved I, Q) across the planned files, in order."""

    def __init__(self, files: list[OutFile], fmt: str, fs: int) -> None:
        self.files, self.fmt, self.fs = files, fmt, fs
        self._i = -1
        self._f: BinaryIO | None = None
        self._left = 0  # samples still to write into the current file
        self.written = 0

    def _next(self) -> None:
        self._close_current()
        self._i += 1
        if self._i >= len(self.files):
            raise ValueError("more samples than planned")
        of = self.files[self._i]
        of.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = of.path.open("wb")
        if self.fmt == "wav":
            self._f.write(wav_header(of.n_samples, self.fs))
        elif self.fmt == "rf64":
            self._f.write(rf64_header(of.n_samples, self.fs))
        self._left = of.n_samples

    def write(self, u8: np.ndarray) -> None:
        buf = memoryview(u8.tobytes())
        while len(buf):
            if self._left == 0:
                self._next()
            assert self._f is not None
            take = min(self._left, len(buf) // 2)
            self._f.write(buf[: 2 * take])
            buf = buf[2 * take :]
            self._left -= take
            self.written += take

    def _close_current(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def close_quietly(self) -> None:
        """Close after an error, leaving what was written (its header sizes stay too large)."""
        self._close_current()

    def close(self) -> None:
        self._close_current()
        total = sum(f.n_samples for f in self.files)
        if self.written != total:
            raise ValueError(f"wrote {self.written} samples, planned {total}")
