"""Ground and local platforms: transmitters without geometry.

- `ground`: a terrestrial transmitter (VDL ground station, airband AM, the 138 MHz carrier, and aircraft VDL bursts,
  which get a random frequency offset and level per burst instead of a flight geometry).
- `local`: interference inside the receiver (birdies, USB noise humps); the only platform wideband signals sit on.
Both are always "visible" and have a fixed received level (`level: {cn0_dbhz | rx_dbm}`); there is no Doppler, no
antenna pattern and no catalog entry.
"""

from typing import TYPE_CHECKING, override

from assist.platforms.base import Platform, PlatformContext, register_platform

if TYPE_CHECKING:
    import numpy as np

    from assist.orbits.geometry import Geometry, Pass


class FixedPlatform(Platform):
    fixed = True

    def __init__(self, source_id: str, name: str) -> None:
        self.source_id = source_id
        self.name = name

    @override
    @classmethod
    def build(cls, ctx: PlatformContext) -> FixedPlatform:
        return cls(ctx.source_id, ctx.name)

    @override
    def geometry(self, t_unix: np.ndarray) -> Geometry:
        raise ValueError(f"source {self.source_id}: a {self.kind} platform has no geometry")

    @override
    def passes(self, t0: float, t1: float) -> list[Pass]:
        return []


@register_platform("ground")
class GroundPlatform(FixedPlatform):
    pass


@register_platform("local")
class LocalPlatform(FixedPlatform):
    allows_wideband = True
