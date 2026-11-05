"""Carrier model (plan §6.3).

f_tx(t) = f_nominal + offset_hz + pass_offset(p) + drift(p) * (t - t_aos(p))

Frequencies are quantised to 1 mHz and kept as integers, so the large part of a frequency is exact; only the
drift term is a (small) float. t_aos(p) is the pass key: the first whole Unix second at or above the horizon.
Per-pass draws use the stream (source_id, emitter_id, "pass", aos_key): pass_offset ~ normal(0, sigma) first,
then the drift sign (+ or - with equal probability).
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from assist.rng import KeyedRng


def to_mhz(f_hz: float) -> int:
    """Frequency in integer millihertz (the carrier quantum)."""
    return round(f_hz * 1000)


@dataclass
class Carrier:
    source_id: str
    emitter_id: str
    rest_mhz: int  # nominal + fixed offset, mHz
    pass_offset_sigma_hz: float | None = None  # zombie oscillator; None: no per-pass terms
    drift_hz_s: float | None = None
    rng: KeyedRng | None = None
    _cache: dict[int, tuple[int, float]] = field(default_factory=dict, repr=False)

    @property
    def per_pass(self) -> bool:
        return self.pass_offset_sigma_hz is not None or self.drift_hz_s is not None

    def pass_terms(self, aos_key: int) -> tuple[int, float]:
        """(pass offset in mHz, signed drift in Hz/s) for the pass with this key."""
        if not self.per_pass:
            return 0, 0.0
        hit = self._cache.get(aos_key)
        if hit is None:
            if self.rng is None:
                raise ValueError("per-pass carrier terms need an RNG")
            g = self.rng.stream(self.source_id, self.emitter_id, "pass", aos_key)
            offset = to_mhz(float(g.normal(0.0, self.pass_offset_sigma_hz or 0.0)))
            sign = -1.0 if g.random() < 0.5 else 1.0
            hit = (offset, sign * (self.drift_hz_s or 0.0))
            self._cache[aos_key] = hit
        return hit

    def f_tx(self, t_unix: np.ndarray, aos_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(base in integer mHz, drift part in Hz) so that f_tx = base / 1000 + drift part.

        aos_keys: the pass key of each time (-1 where no pass; those get the rest frequency).
        """
        t = np.asarray(t_unix, dtype=np.float64)
        base = np.full(t.shape, self.rest_mhz, dtype=np.int64)
        small = np.zeros(t.shape)
        if self.per_pass:
            for k in np.unique(aos_keys):
                if k < 0:
                    continue
                m = aos_keys == k
                off, drift = self.pass_terms(int(k))
                base[m] += off
                small[m] = drift * (t[m] - float(k))
        return base, small
