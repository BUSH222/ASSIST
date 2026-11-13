"""KeyedRng and the time base."""

import datetime as dt

import numpy as np
import pytest

from assist.rng import KeyedRng, RngScope, key_int
from assist.signals.base import Keys
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


def test_scope_prefixes_keys() -> None:
    """A scope's draws are the raw draws with its prefix in front (so moving a consumer to a scope changes nothing)."""
    rng = KeyedRng(42)
    sc = rng.scope("src", "em")
    assert sc.prefix == ("src", "em")
    assert sc.key64("sym") == rng.key64("src", "em", "sym")
    assert sc.key64("comb", 7) == rng.key64("src", "em", "comb", 7)
    np.testing.assert_array_equal(sc.stream("slot", 3).random(4), rng.stream("src", "em", "slot", 3).random(4))
    assert sc.child("a").key64("b") == sc.key64("a", "b") == rng.scope("src").child("em", "a").key64("b")
    assert RngScope(rng).key64("noise") == rng.key64("noise")


def test_scope_is_deterministic() -> None:
    a, b = KeyedRng(5).scope("signal", "msk", "s", "e"), KeyedRng(5).scope("signal", "msk", "s", "e")
    assert a.key64("burst") == b.key64("burst")
    np.testing.assert_array_equal(a.stream("ev", 10).random(3), b.stream("ev", 10).random(3))
    assert a.key64("burst") != KeyedRng(6).scope("signal", "msk", "s", "e").key64("burst")
    assert a.stream("ev", 10).random() != a.stream("ev", 11).random()


def test_scopes_do_not_collide() -> None:
    """Two consumers using the same purpose name under different prefixes get different keys and streams."""
    rng = KeyedRng(9)
    scopes = [
        rng.scope("signal", "msk", "s", "e"),
        rng.scope("signal", "psk8", "s", "e"),
        rng.scope("sink", "msk", "s", "e"),
        rng.scope("signal", "msk", "s", "e2"),
        rng.scope("s", "e"),  # a core emitter's unprefixed scope
    ]
    keys = [sc.key64("burst") for sc in scopes]
    assert len(set(keys)) == len(keys)
    draws = [float(sc.stream("burst", 0).random()) for sc in scopes]
    assert len(set(draws)) == len(draws)


def test_scope_hides_the_rng_and_checks_keys() -> None:
    sc = KeyedRng(1).scope("a")
    assert not hasattr(sc, "rng") and not hasattr(sc, "seed")
    assert repr(sc) == "RngScope('a')"
    with pytest.raises(ValueError):
        KeyedRng(1).scope("a", -1)
    with pytest.raises(TypeError):
        sc.child(1.5)  # ty: ignore[invalid-argument-type]


# key64 of (source "src", emitter "em", purpose) at seed 11, per core signal kind: these numbers address every
# payload bit, burst symbol and noise sample, so they must never change.
CORE_KEYS = [
    ("oqpsk", ("sym",), 2734349363913181313),
    ("oqpsk", ("frame",), 7294017533489003585),
    ("oqpsk_burst", ("segment",), 10425431688845281425),
    ("oqpsk_burst", ("block",), 7155523088879636571),
    ("sdpsk", ("sym",), 2734349363913181313),
    ("lora", ("burst",), 4982388422282931279),
    ("d8psk", ("burst",), 4982388422282931279),
    ("gfsk", ("burst",), 4982388422282931279),
    ("am", ("audio",), 1157163635034467531),
    ("noise_band", ("noise_band",), 5372446142627992446),
    ("noise_comb", ("comb", 1234567), 17735049748196678244),
]


@pytest.mark.parametrize(("kind", "parts", "want"), CORE_KEYS)
def test_core_kind_keys_are_pinned(kind: str, parts: tuple[str | int, ...], want: int) -> None:
    assert Keys(KeyedRng(11), "src", "em").key64(*parts) == want


def test_core_kind_streams_are_pinned() -> None:
    """The two numpy streams of core kinds: am's syllables and noise_comb's wander, keyed by the event start (us)."""
    keys = Keys(KeyedRng(11), "src", "em")
    assert (keys.source_id, keys.emitter_id, keys.prefix) == ("src", "em", ("src", "em"))
    assert keys.stream("syllables", 1234567).integers(0, 2**32) == 2748386301
    assert keys.stream("wander", 1234567).integers(0, 2**32) == 2528763046
