"""P1: cu8 WAV reader (RIFF, RF64, wrapped size fields, auxi) and SDR++ start times."""

import datetime as dt
import struct
from typing import TYPE_CHECKING

import numpy as np
import pytest

from assist.profile.wavio import Recording, parse_tz, read_wav_info, sdrpp_name_info

if TYPE_CHECKING:
    from pathlib import Path

NAME = "baseband_137000000Hz_19-39-00_01-11-2026.wav"
FS = 2_400_000


def _fmt(bits: int = 8, channels: int = 2, tag: int = 1) -> bytes:
    block = channels * bits // 8
    return b"fmt " + struct.pack("<IHHIIHH", 16, tag, channels, FS, FS * block, block, bits)


def _chunk(cid: bytes, body: bytes) -> bytes:
    return cid + struct.pack("<I", len(body)) + body + (b"\0" if len(body) % 2 else b"")


def _write_riff(path: Path, data: bytes, pre: bytes = b"", post: bytes = b"", data_size: int | None = None) -> None:
    size = len(data) if data_size is None else data_size
    body = b"WAVE" + _fmt() + pre + b"data" + struct.pack("<I", size) + data + post
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body) & 0xFFFFFFFF) + body)


def test_sdrpp_name_gives_utc_start() -> None:
    start, fc = sdrpp_name_info(NAME, "+03:00")
    assert start == dt.datetime(2026, 11, 1, 16, 39, 0, tzinfo=dt.UTC)
    assert fc == 137_000_000


@pytest.mark.parametrize("bad", ["recording.wav", "baseband_137000000Hz_19-39_01-11-2026.wav"])
def test_sdrpp_name_rejects_other_names(bad: str) -> None:
    with pytest.raises(ValueError):
        sdrpp_name_info(bad, "+03:00")


def test_parse_tz() -> None:
    assert parse_tz("+03:00").utcoffset(None) == dt.timedelta(hours=3)
    assert parse_tz("-0530").utcoffset(None) == -dt.timedelta(hours=5, minutes=30)
    assert parse_tz("UTC").utcoffset(None) == dt.timedelta(0)
    with pytest.raises(ValueError):
        parse_tz("Europe/Moscow")


def test_plain_riff(tmp_path: Path) -> None:
    p = tmp_path / NAME
    data = bytes(range(200))
    _write_riff(p, data, post=_chunk(b"LIST", b"INFOtest"))  # trailing chunk: declared size must be used
    info = read_wav_info(p)
    assert (info.fs, info.data_offset, info.n_samples) == (FS, 44, 100)
    assert not info.rf64
    assert not info.size_fields_wrong


def test_auxi_chunk(tmp_path: Path) -> None:
    systime = struct.pack("<8H", 2026, 11, 0, 1, 19, 39, 0, 250)
    auxi = _chunk(b"auxi", systime + systime + struct.pack("<I", 137_000_000))
    p = tmp_path / NAME
    _write_riff(p, bytes(20), pre=auxi)
    info = read_wav_info(p)
    assert info.auxi is not None
    assert info.auxi["center_freq_hz"] == 137_000_000
    assert info.auxi["start_time"] == dt.datetime(2026, 11, 1, 19, 39, 0, 250_000)
    assert info.n_samples == 10


def test_truncated_file_reads_to_eof(tmp_path: Path) -> None:
    p = tmp_path / NAME
    _write_riff(p, bytes(64), data_size=1000)
    info = read_wav_info(p)
    assert info.size_fields_wrong
    assert info.n_samples == 32


def test_rf64(tmp_path: Path) -> None:
    data = bytes(range(64))
    ds64 = _chunk(b"ds64", struct.pack("<QQQI", 0, len(data), len(data) // 2, 0))
    body = b"WAVE" + ds64 + _fmt() + b"data" + struct.pack("<I", 0xFFFFFFFF) + data
    p = tmp_path / NAME
    p.write_bytes(b"RF64" + struct.pack("<I", 0xFFFFFFFF) + body)
    info = read_wav_info(p)
    assert info.rf64
    assert not info.size_fields_wrong
    assert info.n_samples == 32
    assert info.declared_data_bytes == 64


def test_over_4gib_with_wrapped_sizes(tmp_path: Path) -> None:
    """Like the station's recordings: RIFF and data sizes wrapped mod 2^32. Sparse file, no real disk use."""
    n_bytes = 2**32 + 1000
    p = tmp_path / NAME
    _write_riff(p, b"", data_size=n_bytes & 0xFFFFFFFF)
    with p.open("r+b") as f:
        f.seek(44 + n_bytes - 2)
        f.write(b"\x80\x7f")  # last sample
    info = read_wav_info(p)
    assert info.size_fields_wrong
    assert info.declared_data_bytes == 1000
    assert info.n_samples == n_bytes // 2
    rec = Recording(p)
    assert rec.u8(rec.n_samples - 1, 5).tolist() == [0x80, 0x7F]


def test_rejects_non_cu8(tmp_path: Path) -> None:
    p = tmp_path / NAME
    body = b"WAVE" + _fmt(bits=16) + b"data" + struct.pack("<I", 8) + bytes(8)
    p.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    with pytest.raises(ValueError, match="8-bit"):
        read_wav_info(p)


def test_recording_timing_and_samples(tmp_path: Path) -> None:
    p = tmp_path / NAME
    u8 = np.arange(40, dtype=np.uint8)
    _write_riff(p, u8.tobytes())
    rec = Recording(p, tz="+03:00")
    assert rec.start_unix == dt.datetime(2026, 11, 1, 16, 39, tzinfo=dt.UTC).timestamp()
    assert rec.time_of(FS) == rec.start_unix + 1.0
    assert rec.index_of(rec.start_unix + 3 / FS) == 3
    assert rec.u8(18, 10).tolist() == [36, 37, 38, 39]  # clipped at the end
    iq = rec.iq(0, 2)
    assert iq.dtype == np.complex64
    i, q = np.array([0, 2], dtype=np.float32), np.array([1, 3], dtype=np.float32)
    np.testing.assert_array_equal(iq, (i - 127.5) / 127.5 + 1j * (q - 127.5) / 127.5)
