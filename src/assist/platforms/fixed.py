"""Ground and local platforms (plan §3.4, M7): transmitters without geometry.

- `ground`: a terrestrial transmitter (VDL ground station, airband AM, the 138 MHz carrier, and aircraft VDL bursts,
  which get a random frequency offset and level per burst instead of a flight geometry).
- `local`: interference inside the receiver (birdies, USB noise humps).
Both are always "visible" and have a fixed received level (`level: {cn0_dbhz | rx_dbm}`); there is no Doppler, no
antenna pattern and no catalog entry.
"""

from typing import TYPE_CHECKING, Literal, override

from assist.platforms.base import Platform

if TYPE_CHECKING:
    import numpy as np

    from assist.config import schema as S
    from assist.orbits.geometry import Geometry, Pass


class FixedPlatform(Platform):
    fixed = True
    norad: int | None = None
    in_catalog = False

    def __init__(self, src: S.Source, kind: Literal["ground", "local"]) -> None:
        self.source_id = src.id
        self.kind = kind
        self.name = src.name or src.id

    @override
    def geometry(self, t_unix: np.ndarray) -> Geometry:
        raise ValueError(f"source {self.source_id}: a {self.kind} platform has no geometry")

    @override
    def passes(self, t0: float, t1: float) -> list[Pass]:
        return []
