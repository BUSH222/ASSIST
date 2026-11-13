"""Keyed randomness (plan §3.2).

Every random draw comes from a stream keyed by stable names (source and emitter ids, absolute time indices), never
by list position or Python's hash(). Adding or removing a source therefore never changes another source's draws.
"""

import hashlib
from typing import override

import numpy as np


def key_int(k: str | int) -> int:
    """Integer form of one key element: strings via blake2b (8 bytes, little endian), ints as themselves."""
    if isinstance(k, bool) or not isinstance(k, str | int):
        raise TypeError(f"RNG key elements must be str or int, not {type(k).__name__}")
    if isinstance(k, int):
        if k < 0:
            raise ValueError(f"RNG key integers must be >= 0, got {k}")
        return k
    return int.from_bytes(hashlib.blake2b(k.encode(), digest_size=8).digest(), "little")


class KeyedRng:
    def __init__(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"seed must be an integer >= 0, got {seed!r}")
        self.seed = seed

    def stream(self, *key: str | int) -> np.random.Generator:
        """Generator(Philox(SeedSequence(seed, spawn_key=h(key)))): the same key always gives the same stream."""
        return np.random.Generator(np.random.Philox(self._seq(key)))

    def key64(self, *key: str | int) -> int:
        """A 64-bit key from the same seed sequence, for counter-based generators outside numpy (the C++ noise,
        whose counter is the absolute hop index, plan §3.2)."""
        return int(self._seq(key).generate_state(1, np.uint64)[0])

    def _seq(self, key: tuple[str | int, ...]) -> np.random.SeedSequence:
        return np.random.SeedSequence(self.seed, spawn_key=tuple(key_int(k) for k in key))

    def scope(self, *prefix: str | int) -> RngScope:
        """The part of the key space under `prefix`."""
        return RngScope(self, prefix)


class RngScope:
    """One consumer's share of the scenario's randomness: every key it uses starts with its prefix.

    A scope hands out streams and 64-bit keys but not the seed or the `KeyedRng`, so a consumer cannot reach another
    consumer's draws. Two scopes whose prefixes differ never give the same key, whatever names are used below them.

    - `stream(*parts)`: a numpy Generator, for a few draws made in one go (an event's parameters, a per-pass offset).
      Put the absolute index of the thing drawn (event start, slot, pass) in `parts`.
    - `key64(*parts)`: a key for the counter-based kernels (`random_bytes`, `white`), whose counter is an absolute
      index, so any window of a long sequence is generated on its own.
    """

    __slots__ = ("_rng", "prefix")

    def __init__(self, rng: KeyedRng, prefix: tuple[str | int, ...] = ()) -> None:
        for k in prefix:
            key_int(k)  # reject bad key elements here, not at the first draw
        self._rng = rng
        self.prefix = tuple(prefix)

    def stream(self, *parts: str | int) -> np.random.Generator:
        return self._rng.stream(*self.prefix, *parts)

    def key64(self, *parts: str | int) -> int:
        return self._rng.key64(*self.prefix, *parts)

    def child(self, *parts: str | int) -> RngScope:
        """A narrower scope: this prefix followed by `parts`."""
        return RngScope(self._rng, (*self.prefix, *parts))

    @override
    def __repr__(self) -> str:
        return f"RngScope({', '.join(repr(k) for k in self.prefix)})"
