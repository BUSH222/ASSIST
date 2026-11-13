"""cu8 WAV reader.

Reads RIFF and RF64 files holding interleaved unsigned 8-bit I/Q, memory-mapped. RIFF size fields of files over
4 GiB are wrong (SDR++ lets them wrap mod 2^32); in that case the data chunk runs to the end of the file. The start
time comes from the SDR++ file name, which is in station local time.
"""

import datetime as dt
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from assist.timebase import parse_tz
from assist.waterfall import iq_from_u8

__all__ = ["Recording", "WavInfo", "parse_tz", "read_wav_info", "sdrpp_name_info"]

_U32_MAX = 0xFFFFFFFF
_FORMAT_PCM = 1
_FORMAT_EXTENSIBLE = 0xFFFE

# baseband_137000000Hz_19-39-00_01-11-2026.wav -> fc, HH-MM-SS, DD-MM-YYYY (local time)
_SDRPP_NAME = re.compile(
    r"baseband_(?P<fc>\d+)Hz_(?P<H>\d{2})-(?P<M>\d{2})-(?P<S>\d{2})_(?P<d>\d{2})-(?P<m>\d{2})-(?P<Y>\d{4})"
)


def sdrpp_name_info(name: str, tz: str) -> tuple[dt.datetime, int]:
    """(start time in UTC, centre frequency in Hz) from an SDR++ baseband file name in local time `tz`."""
    m = _SDRPP_NAME.search(name)
    if not m:
        raise ValueError(f"not an SDR++ baseband file name: {name!r}")
    g = {k: int(v) for k, v in m.groupdict().items()}
    local = dt.datetime(g["Y"], g["m"], g["d"], g["H"], g["M"], g["S"], tzinfo=parse_tz(tz))
    return local.astimezone(dt.UTC), g["fc"]


@dataclass(frozen=True)
class WavInfo:
    path: Path
    fs: int
    data_offset: int  # byte offset of the first sample
    n_samples: int  # complex samples
    declared_data_bytes: int  # size field of the data chunk (or ds64), as written
    rf64: bool
    size_fields_wrong: bool  # data chunk taken to EOF because its size field did not fit the file
    auxi: dict[str, object] | None  # SDR#/HDSDR `auxi` chunk, if present


def _systemtime(b: bytes) -> dt.datetime | None:
    year, month, _dow, day, hour, minute, second, ms = struct.unpack("<8H", b)
    try:
        return dt.datetime(year, month, day, hour, minute, second, ms * 1000)  # naive: the writer's clock
    except ValueError:
        return None


def _parse_auxi(b: bytes) -> dict[str, object]:
    out: dict[str, object] = {}
    if len(b) >= 32:
        out["start_time"] = _systemtime(b[0:16])
        out["stop_time"] = _systemtime(b[16:32])
    if len(b) >= 36:
        out["center_freq_hz"] = struct.unpack("<I", b[32:36])[0]
    return out


def read_wav_info(path: str | Path) -> WavInfo:
    """Parse the chunk structure of an 8-bit, 2-channel PCM WAV (RIFF or RF64)."""
    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as f:
        head = f.read(12)
        if len(head) < 12 or head[8:12] != b"WAVE" or head[0:4] not in {b"RIFF", b"RF64"}:
            raise ValueError(f"{path.name}: not a RIFF/RF64 WAVE file")
        rf64 = head[0:4] == b"RF64"
        ds64_data: int | None = None
        fmt: tuple[int, int, int, int] | None = None
        auxi: dict[str, object] | None = None
        pos = 12
        while pos + 8 <= file_size:
            f.seek(pos)
            cid, size = struct.unpack("<4sI", f.read(8))
            body = pos + 8
            if cid == b"ds64":
                _riff64, ds64_data = struct.unpack("<QQ", f.read(16))
            elif cid == b"fmt ":
                tag, channels, fs, _byte_rate, _align, bits = struct.unpack("<HHIIHH", f.read(16))
                fmt = (tag, channels, fs, bits)
            elif cid == b"auxi":
                auxi = _parse_auxi(f.read(min(size, 4096)))
            elif cid == b"data":
                if fmt is None:
                    raise ValueError(f"{path.name}: data chunk before fmt chunk")
                tag, channels, fs, bits = fmt
                if tag not in {_FORMAT_PCM, _FORMAT_EXTENSIBLE} or channels != 2 or bits != 8:
                    raise ValueError(
                        f"{path.name}: expected 8-bit 2-channel PCM, got tag={tag} ch={channels} bits={bits}"
                    )
                remaining = file_size - body
                declared = ds64_data if (rf64 and size == _U32_MAX and ds64_data is not None) else size
                # A RIFF size field cannot describe more than 4 GiB: if the declared size doesn't fit the file
                # (wrapped, zero, or past EOF), the data runs to the end of the file.
                wrong = declared > remaining or (not rf64 and remaining > _U32_MAX and declared != remaining)
                n_bytes = remaining if wrong or declared == 0 else declared
                return WavInfo(path, fs, body, n_bytes // 2, declared, rf64, wrong, auxi)
            pos = body + size + (size & 1)  # chunks are word-aligned
    raise ValueError(f"{path.name}: no data chunk")


class Recording:
    """A cu8 WAV recording: memory-mapped samples plus absolute timing.

    Sample `n` is at `start_unix + n / fs`. Reads are by absolute sample index and never load the whole file.
    """

    def __init__(self, path: str | Path, tz: str = "+03:00", start_utc: dt.datetime | None = None) -> None:
        self.info = read_wav_info(path)
        self.path = self.info.path
        self.fs = self.info.fs
        self.n_samples = self.info.n_samples
        if start_utc is None:
            start_utc, fc = sdrpp_name_info(self.path.name, tz)
            self.fc_hz: int | None = fc
        else:
            self.fc_hz = None
        self.start_utc = start_utc
        self.start_unix = start_utc.timestamp()
        self._u8 = np.memmap(
            self.path, dtype=np.uint8, mode="r", offset=self.info.data_offset, shape=(self.n_samples * 2,)
        )

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs

    def time_of(self, n: int | np.ndarray) -> float | np.ndarray:
        """Unix time (UTC) of sample index n."""
        return self.start_unix + np.asarray(n, dtype=np.float64) / self.fs

    def index_of(self, t_unix: float) -> int:
        """Sample index at (or just after) unix time t."""
        return int(np.ceil((t_unix - self.start_unix) * self.fs))

    def u8(self, n0: int, n: int) -> np.ndarray:
        """Raw interleaved I,Q bytes for samples [n0, n0+n), clipped to the file. Shape (2*m,), uint8, a copy."""
        n0 = max(0, n0)
        n1 = min(self.n_samples, n0 + n)
        return np.array(self._u8[2 * n0 : 2 * max(n0, n1)])

    def iq(self, n0: int, n: int) -> np.ndarray:
        """complex64 samples scaled like the capture script: (u - 127.5) / 127.5."""
        return iq_from_u8(self.u8(n0, n))
