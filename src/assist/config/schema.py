"""Scenario schema (plan §3.3, §3.4, §6, Appendix A): pydantic models of a fully merged scenario.

This is the document after includes, `--set`, band defaults, presets and random expressions have been applied; band
defaults fill `frontend` and `antenna`. Unknown keys are errors (typos fail loudly). `assist schema` exports it.
"""

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    field_validator,
    model_validator,
)

from assist.emitters.link import QFH_COEFFS_DB
from assist.signals.base import get_kind
from assist.station import DEFAULT_TZ, FALLBACK_ALT_M, FALLBACK_LAT_DEG, FALLBACK_LON_DEG
from assist.timebase import format_iso, parse_duration, parse_iso


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Clock(_M):
    start: str = Field("now", description="ISO 8601 time with zone (e.g. 2026-11-01T16:39:00Z), or `now`.")
    duration: float = Field(3600.0, description="Seconds; strings like 90s, 10min, 2h, 1d are accepted.")

    @field_validator("start")
    @classmethod
    def _start(cls, v: str) -> str:
        return v if v == "now" else format_iso(parse_iso(v))

    @field_validator("duration", mode="before")
    @classmethod
    def _duration(cls, v: Any) -> float:
        d = parse_duration(v)
        if d <= 0:
            raise ValueError("duration must be > 0")
        return d


class StationConfig(_M):
    file: str | None = Field(
        None,
        description="Station file (station_lat, station_lon, altitude_m). Default: samples/station.yaml when "
        "neither file nor lat/lon is given; the rounded fallback position is used when it is absent.",
    )
    lat: float = FALLBACK_LAT_DEG
    lon: float = FALLBACK_LON_DEG
    alt_m: float = FALLBACK_ALT_M
    horizon_deg: float = 0.0
    tz: str = Field(DEFAULT_TZ, description="Station local time zone, used only for SDR++-style file names.")
    source: str | None = Field(None, description="Set by resolve: where the position came from.")


class Lna(_M):
    gain_db: float
    nf_db: float


class Frontend(_M):
    fc_hz: float
    fs_hz: float = Field(description="Sample rate; an integer multiple of 100 Hz (10 ms hops).")
    tuner_gain_db: float
    ppm: float = Field(description="Crystal error (sample clock and LO), plan §3.1. Only 0 is supported until M7.")
    lna: Lna
    tuner_nf_db: float
    saw_loss_db: float = Field(description="SAW loss used for T_sys (scalar until the SAW table arrives in M7).")
    t_ant_k: float
    image_db: float
    iip3_dbm: float
    noise_sigma_counts: float | None = Field(
        None, gt=0, description="ADC noise sigma per I/Q component (counts). Default: the profile's, else the band's."
    )

    @field_validator("ppm")
    @classmethod
    def _ppm(cls, v: float) -> float:
        if v != 0:
            raise ValueError(
                "only 0 is supported until M7 (the crystal error of the stream clock is not modelled yet)"
            )
        return v

    @field_validator("fs_hz")
    @classmethod
    def _fs(cls, v: float) -> float:
        if v <= 0 or v != int(v) or int(v) % 100:
            raise ValueError("fs_hz must be a positive integer multiple of 100 Hz")
        return v


def _pattern_alias(v: Any) -> Any:
    if v == "qfh_poly":
        return {"type": "poly", "coeffs_db": list(QFH_COEFFS_DB)}
    if v == "isotropic":
        return {"type": "isotropic"}
    return v


class PolyPattern(_M):
    type: Literal["poly"]
    coeffs_db: list[float]
    max_deg: float = Field(90.0, description="Outside 0..max_deg the gain is clamped to its max_deg value.")


class CardioidPattern(_M):
    type: Literal["cardioid"]
    n: float
    peak_dbi: float = 0.0


class IsotropicPattern(_M):
    type: Literal["isotropic"]
    gain_dbi: float = 0.0


PatternSpec = Annotated[
    PolyPattern | CardioidPattern | IsotropicPattern, Field(discriminator="type"), BeforeValidator(_pattern_alias)
]


class Antenna(_M):
    pattern: PatternSpec


class Elements(_M):
    files: list[str] = Field(default_factory=list, description="OMM CSV/JSON or TLE files (paths from the cwd).")


class Perturb(_M):
    along_track_s: float = Field(0.0, description="> 0: the true satellite is this many seconds behind.")
    raan_deg: float = 0.0
    inc_deg: float = 0.0
    dn_rev_day: float = 0.0


class CatalogOrbit(_M):
    norad: int
    epoch: str | None = Field(None, description="Choose the element set nearest this time (default: clock start).")
    perturb: Perturb | None = None
    omm: dict[str, Any] | None = Field(None, description="Set by resolve: the published element set used.")

    @field_validator("epoch")
    @classmethod
    def _epoch(cls, v: str | None) -> str | None:
        return None if v is None else format_iso(parse_iso(v))


class KeplerElements(_M):
    epoch: str
    alt_km: float | None = None
    a_km: float | None = None
    e: float = 0.0
    i_deg: float
    raan_deg: float = 0.0
    argp_deg: float = 0.0
    m_deg: float = 0.0
    bstar: float = 0.0

    @field_validator("epoch")
    @classmethod
    def _epoch(cls, v: str) -> str:
        return format_iso(parse_iso(v))

    @model_validator(mode="after")
    def _one_size(self) -> KeplerElements:
        if (self.alt_km is None) == (self.a_km is None):
            raise ValueError("give exactly one of alt_km and a_km")
        return self


class KeplerOrbit(_M):
    kepler: KeplerElements


class PassSolution(_M):
    raan_deg: float
    m_deg: float


class SyntheticPass(_M):
    aos: str = Field(description="ISO 8601 time, or an offset from the clock start such as +2min.")
    max_el_deg: float = Field(gt=0, le=90)
    direction: Literal["asc", "desc"] | None = Field(None, description="Default: drawn per source (keyed).")
    side: Literal["east", "west"] | None = Field(
        None, description="Sky half of the culmination. Default: drawn per source (keyed)."
    )
    alt_km: float = 820.0
    inc_deg: float = 98.7
    solution: PassSolution | None = Field(None, description="Set by resolve: the solved orbit.")


class SyntheticPassOrbit(_M):
    synthetic_pass: SyntheticPass


def _orbit_tag(v: Any) -> str | None:
    if isinstance(v, dict):
        for k in ("norad", "kepler", "synthetic_pass"):
            if k in v:
                return k
        return None
    return {CatalogOrbit: "norad", KeplerOrbit: "kepler", SyntheticPassOrbit: "synthetic_pass"}.get(type(v))


Orbit = Annotated[
    Annotated[CatalogOrbit, Tag("norad")]
    | Annotated[KeplerOrbit, Tag("kepler")]
    | Annotated[SyntheticPassOrbit, Tag("synthetic_pass")],
    Discriminator(
        _orbit_tag, custom_error_type="orbit", custom_error_message="orbit needs norad, kepler or synthetic_pass"
    ),
]


class SignalConfig(_M):
    kind: str
    params: dict[str, Any] = Field(default_factory=dict, description="Checked once the signal kind is implemented.")

    @field_validator("kind")
    @classmethod
    def _kind(cls, v: str) -> str:
        get_kind(v)
        return v


class Oscillator(_M):
    pass_offset_sigma_hz: float = Field(50.0, ge=0)
    drift_hz_s: float = Field(0.2, ge=0, description="Magnitude; the sign is drawn per pass.")


class ContinuousSchedule(_M):
    type: Literal["continuous"] = "continuous"


class OnOffSchedule(_M):
    type: Literal["on_off"]
    on_s: float = Field(40.0, gt=0)
    off_s: float = Field(180.0, ge=0)
    phase_s: float | None = Field(None, description="Start of an on interval (mod period). Default: keyed draw.")
    ramp_ms: float = Field(5.0, ge=0)


class Duration(_M):
    dist: Literal["fixed", "lognormal", "exponential"] = "fixed"
    s: float = Field(1.0, ge=0)
    median_s: float = Field(1.0, gt=0)
    sigma: float = Field(0.5, ge=0)
    mean_s: float = Field(1.0, gt=0)
    max_s: float = Field(3600.0, gt=0)


class ExpGapsSchedule(_M):
    type: Literal["exp_gaps"]
    mean_gap_s: float = Field(180.0, gt=0)
    min_gap_s: float = Field(20.0, ge=0)
    duration: Duration = Field(default_factory=Duration)


class PoissonSchedule(_M):
    type: Literal["poisson_events"]
    rate_per_hour: float | list[float] = Field(description="Constant, or 24 values by UTC hour.")
    duration: Duration = Field(default_factory=Duration)

    @field_validator("rate_per_hour")
    @classmethod
    def _rate(cls, v: float | list[float]) -> float | list[float]:
        if isinstance(v, list) and len(v) != 24:
            raise ValueError("rate_per_hour needs 24 hourly values")
        return v


ScheduleConfig = Annotated[
    ContinuousSchedule | OnOffSchedule | ExpGapsSchedule | PoissonSchedule, Field(discriminator="type")
]


class AmConfig(_M):
    period_s: float = Field(gt=0)
    depth_db: float = Field(ge=0)
    phase_rad: float | None = Field(None, description="Default: keyed draw.")


class Level(_M):
    p_tx_dbw: float | None = Field(None, description="Power into the transmit antenna.")
    cn0_zenith_dbhz: float | None = Field(None, description="C/N0 of an overhead pass; solved to p_tx_dbw.")
    cal: str | None = Field(None, description="Calibrated level from the profile or band (e.g. meteor_m2_3).")
    extra_loss_db: float = 0.0
    resolved_from: str | None = Field(None, description="Set by resolve: how p_tx_dbw was obtained.")

    @model_validator(mode="after")
    def _one(self) -> Level:
        given = [k for k in ("p_tx_dbw", "cn0_zenith_dbhz", "cal") if getattr(self, k) is not None]
        if self.resolved_from is not None:
            if given != ["p_tx_dbw"]:
                raise ValueError("a resolved level holds only p_tx_dbw")
        elif len(given) != 1:
            raise ValueError("give exactly one of p_tx_dbw, cn0_zenith_dbhz, cal")
        return self


class Emitter(_M):
    id: str = Field(min_length=1)
    enabled: bool = True
    signal: SignalConfig
    freq_hz: float = Field(gt=0, description="Nominal frequency (quantised to 1 mHz at resolve).")
    offset_hz: float = Field(0.0, description="Fixed offset (rest-frequency error).")
    oscillator: Oscillator | None = Field(None, description="Per-pass offset and drift (zombies).")
    schedule: ScheduleConfig = Field(default_factory=ContinuousSchedule)
    am: AmConfig | None = None
    level: Level
    pattern: PatternSpec = Field(default_factory=lambda: PolyPattern(type="poly", coeffs_db=list(QFH_COEFFS_DB)))


class Source(_M):
    id: str = Field(min_length=1)
    preset: str | None = Field(None, description="Preset the source was expanded from.")
    platform: Literal["satellite"] = "satellite"
    name: str | None = None
    orbit: Orbit
    in_catalog: bool | None = Field(None, description="Default: true for catalog orbits, false for synthetic ones.")
    catalog_norad: int | None = Field(None, description="NORAD id of a synthetic object in the catalog output.")
    emitters: list[Emitter] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_emitters(self) -> Source:
        ids = [e.id for e in self.emitters]
        if len(ids) != len(set(ids)):
            raise ValueError(f"source {self.id}: duplicate emitter ids {ids}")
        return self


class Scenario(_M):
    version: Literal[1] = 1
    seed: int = Field(0, ge=0)
    clock: Clock = Field(default_factory=Clock)
    station: StationConfig = Field(default_factory=StationConfig)
    band: str = "vhf137"
    profile: str | None = Field(None, description="Profile directory (e.g. profiles/vhf137); band defaults otherwise.")
    frontend: Frontend
    antenna: Antenna
    elements: Elements = Field(default_factory=Elements)
    sources: list[Source] = Field(default_factory=list)
    sinks: list[dict[str, Any]] = Field(default_factory=list, description="Output sinks (used from M3/M4).")
    assist_version: str | None = Field(None, description="Set by resolve.")
    band_hash: str | None = Field(None, description="Set by resolve: sha256 of the band directory.")
    profile_hash: str | None = Field(None, description="Set by resolve: sha256 of the profile (without bank/).")

    @model_validator(mode="after")
    def _unique_sources(self) -> Scenario:
        ids = [s.id for s in self.sources]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"duplicate source ids {dup}")
        return self
