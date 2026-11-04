"""Platform interface (plan §3.4). M2 has satellites; M7 adds ground and local platforms (no geometry, `fixed.py`).

Aircraft are not modelled as platforms (M7 decision): aircraft VDL bursts are ground events with a random frequency
offset and level each."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from assist.orbits.geometry import Geometry, Pass


class Platform(ABC):
    source_id: str
    fixed: bool = False  # ground / local: no geometry, always visible

    @abstractmethod
    def geometry(self, t_unix: np.ndarray) -> Geometry:
        """Range, range rate, elevation, azimuth and off-nadir angle at absolute times (float64)."""

    @abstractmethod
    def passes(self, t0: float, t1: float) -> list[Pass]:
        """Visibility windows overlapping [t0, t1)."""
