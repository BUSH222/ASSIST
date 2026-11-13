"""Platform interface (plan §3.4, PLAN.md R2): where an emitter is and how it moves.

A platform is a class registered under a name with `@register_platform("name")`; a source picks one with
`platform: name`. The core ones are `satellite` (an orbit and the link budget), `ground` and `local` (no geometry,
a fixed received level; `fixed.py`). External ones come from modules a scenario lists under `plugins:`.

Rules about what a source on a platform may have are not tied to names: a platform class declares its
**capabilities** (the class attributes below), and the schema, the resolver and the scene look those up.

Aircraft are not modelled as platforms: aircraft VDL bursts are ground events with a random frequency offset and
level each.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict

from assist import registry

if TYPE_CHECKING:
    from assist.orbits.geometry import Geometry, Pass
    from assist.rng import RngScope
    from assist.station import Station


class PlatformParams(BaseModel):
    """Base of every platform's parameter model (`platform_params:` of a source): unknown keys are errors."""

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class PlatformContext:
    """What a platform is built from."""

    source_id: str
    name: str  # the source's `name`, else its id
    params: PlatformParams  # the validated `platform_params`
    station: Station
    horizon_deg: float
    scope: RngScope  # this platform's randomness: keys under ("platform", platform name, source_id)
    source: Any  # the resolved source (assist.config.schema.Source); core platforms read `orbit` from it


class Platform(ABC):
    kind: ClassVar[str]
    Params: ClassVar[type[PlatformParams]] = PlatformParams

    # --- capabilities ---
    fixed: ClassVar[bool] = False  # no geometry: always visible, no Doppler; `geometry` and `passes` are not used
    needs_orbit: ClassVar[bool] = False  # the source must have an `orbit` (else it must not)
    catalog: ClassVar[bool] = False  # the source may be in the element catalog (`in_catalog`, `catalog_norad`)
    supports_oscillator: ClassVar[bool] = False  # emitters may have per-pass oscillator terms
    # How an emitter's level is given: "link_budget" (p_tx_dbw / cn0_zenith_dbhz / cal, through range and antenna
    # patterns; the core satellite only) or "received" (cn0_dbhz / rx_dbm: the level at the receiver).
    level_mode: ClassVar[Literal["link_budget", "received"]] = "received"
    fades_at_horizon: ClassVar[bool] = False  # `station.fade_s` is applied after AOS and before LOS
    allows_wideband: ClassVar[bool] = False  # wideband (inside-the-receiver) signals may sit on it

    # --- what every platform has ---
    source_id: str
    name: str
    norad: int | None = None
    in_catalog: bool = False

    @classmethod
    @abstractmethod
    def build(cls, ctx: PlatformContext) -> Platform:
        """The platform of one source."""

    @abstractmethod
    def geometry(self, t_unix: np.ndarray) -> Geometry:
        """Range, range rate, elevation, azimuth and off-nadir angle at absolute times (float64). A pure function
        of the time: any window must give the same values."""

    @abstractmethod
    def passes(self, t0: float, t1: float) -> list[Pass]:
        """Visibility windows overlapping [t0, t1), each whole and with an `aos_key` that does not depend on the
        window asked for."""

    def geometry_ms(self, k_ms: np.ndarray) -> Geometry:
        """Geometry at integer Unix milliseconds (the renderer's control grid). Override to form the time from the
        integers when float seconds are not exact enough (they resolve ~0.24 us)."""
        return self.geometry(np.asarray(k_ms, dtype=np.int64) / 1000.0)

    def pass_keys(self, t_unix: np.ndarray, passes: list[Pass]) -> np.ndarray:
        """For each time, the aos_key of the pass containing it, or -1 (not visible)."""
        t = np.asarray(t_unix, dtype=np.float64)
        key = np.full(t.shape, -1, dtype=np.int64)
        for p in passes:
            key[(t >= p.aos) & (t < p.los)] = p.aos_key if p.aos_key is not None else -1
        return key

    def max_speed_km_s(self) -> float:
        """Upper bound on |range rate| (km/s): it sets how much room the emitter's Doppler gets."""
        raise NotImplementedError(f"platform {self.kind}: a platform with geometry must give max_speed_km_s()")


class _Register(Protocol):
    def __call__[C: type[Platform]](self, cls: C, /) -> C: ...


def register_platform(name: str) -> _Register:
    """Class decorator: make a `Platform` subclass available as `platform: name` (core and external alike)."""
    reg = registry.platforms.register(name)

    def deco[C: type[Platform]](cls: C, /) -> C:
        if not (isinstance(cls, type) and issubclass(cls, Platform)):
            raise TypeError(f"register_platform({name!r}) needs a Platform subclass, got {cls!r}")
        if not isinstance(getattr(cls, "Params", None), type) or not issubclass(cls.Params, PlatformParams):
            raise TypeError(f"platform {name!r}: {cls.__name__}.Params must be a PlatformParams subclass")
        if cls.level_mode not in ("link_budget", "received"):
            raise TypeError(f"platform {name!r}: level_mode must be link_budget or received, got {cls.level_mode!r}")
        if not registry.is_core_class(cls) and (cls.level_mode != "received" or cls.needs_orbit or cls.catalog):
            raise TypeError(
                f"platform {name!r}: orbits, the catalog and the link budget belong to the core satellite platform; "
                "an external platform gives its own geometry and takes a received level"
            )
        if cls.fixed and (cls.supports_oscillator or cls.fades_at_horizon):
            raise TypeError(f"platform {name!r}: a fixed platform has no passes (no oscillator terms, no fade)")
        if cls.allows_wideband and not cls.fixed:
            raise TypeError(f"platform {name!r}: wideband signals need a fixed platform")
        setattr(cls, "kind", name)  # noqa: B010 (a ClassVar of the subclass, set through the type variable)
        reg(cls)
        return cls

    return deco


def platform_class(name: str) -> type[Platform]:
    try:
        cls = registry.platforms.get(name)
    except registry.RegistryError as e:
        raise ValueError(str(e)) from None
    assert issubclass(cls, Platform)
    return cls


def check_platform_params(name: str, params: dict[str, Any] | None) -> dict[str, Any]:
    """The platform's parameters validated, with defaults filled in."""
    return platform_class(name).Params.model_validate(params or {}).model_dump(mode="json")
