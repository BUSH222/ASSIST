"""KeyedRng (plan §3.2) and the time base (§3.1)."""

import datetime as dt

import numpy as np
import pytest

from assist.rng import KeyedRng, key_int
from assist.timebase import format_iso, parse_duration, parse_iso, parse_time, unix_to_time, unix_to_time_fast


def test_same_key_same_stream() -> None:
    a = KeyedRng(7).stream("src", "em", "sym", 3).random(5)
    b = KeyedRng(7).stream("src", "em", "sym", 3).random(5)
    np.testing.assert_array_equal(a, b)


def test_keys_and_seeds_separate_streams() -> None:
    base = KeyedRng(7).stream("src", "em", "pass", 100).random()
    assert KeyedRng(7).stream("src", "em", "pass", 101).random() != base
    assert KeyedRng(7).stream("src2", "em", "pass", 100).random() != base
    assert KeyedRng(8).stream("src", "em", "pass", 100).random() != base
    assert KeyedRng(7).stream("em", "src", "pass", 100).random() != base  # order matters


def test_key_hash_is_stable() -> None:
    """blake2b, not Python's (salted) hash(): the value must never change across runs or machines."""
    assert key_int("noise") == int.from_bytes(
        __import__("hashlib").blake2b(b"noise", digest_size=8).digest(), "little"
    )
    assert key_int(5) == 5
    with pytest.raises(ValueError):
        key_int(-1)
    with pytest.raises(TypeError):
        key_int(True)
    with pytest.raises(ValueError):
        KeyedRng(-3)


def test_stream_values_are_pinned() -> None:
    """numpy's Philox stream is part of the reproducibility contract (uv.lock pins numpy)."""
    assert KeyedRng(42).stream("resolve", "z1", "freq_hz").integers(0, 2**32) == 3439137270


def test_iso_round_trip() -> None:
    t = parse_iso("2026-11-01T16:39:00Z")
    assert t == dt.datetime(2026, 11, 1, 16, 39, tzinfo=dt.UTC).timestamp()
    assert parse_iso("2026-11-01T19:39:00+03:00") == t
    assert format_iso(t) == "2026-11-01T16:39:00Z"
    assert format_iso(t + 0.25) == "2026-11-01T16:39:00.250000Z"
    assert format_iso(t + 0.25, 3) == "2026-11-01T16:39:00.250Z"
    assert format_iso(t + 0.9996, 3) == "2026-11-01T16:39:01.000Z"
    with pytest.raises(ValueError, match="zone"):
        parse_iso("2026-11-01T16:39:00")


def test_durations_and_relative_times() -> None:
    assert parse_duration("2h") == 7200
    assert parse_duration("10min") == 600
    assert parse_duration("250ms") == 0.25
    assert parse_duration("1.5e3") == 1500
    assert parse_duration(90) == 90
    with pytest.raises(ValueError):
        parse_duration("2 hours")
    ref = parse_iso("2026-11-01T16:39:00Z")
    assert parse_time("+2min", ref) == ref + 120
    assert parse_time("-30s", ref) == ref - 30
    assert parse_time("start", ref) == ref
    assert parse_time("2026-11-01T16:40:00Z", ref) == ref + 60


def test_fast_time_matches_full_model() -> None:
    t = parse_iso("2026-11-01T16:39:00Z") + np.arange(3.0)
    np.testing.assert_array_equal(unix_to_time(t).tt, unix_to_time_fast(t).tt)


def test_key64_is_stable_and_keyed() -> None:
    """The 64-bit key of the C++ noise generator comes from the same seed sequence as the numpy streams."""
    a, b = KeyedRng(42), KeyedRng(42)
    k = a.key64("noise")
    assert k == b.key64("noise") and 0 <= k < 2**64
    assert k != KeyedRng(43).key64("noise") and k != a.key64("noise2")
    want = np.random.SeedSequence(42, spawn_key=(key_int("noise"),)).generate_state(1, np.uint64)[0]
    assert k == int(want)
