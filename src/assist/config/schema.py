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
from assist.signals.base import check_params, get_kind
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
    fade_s: float = Field(
        15.0,
        ge=0,
        description="Satellite signals rise from pure noise to their link-budget level over this long after AOS and "
        "fall back to noise over this long before LOS (raised-cosine amplitude); there is no horizon map. A pass "
        "shorter than twice this never reaches full level. 0 switches it off.",
    )
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
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters of the kind (plan §5); checked, and defaults filled in, once the kind is implemented.",
    )

    @field_validator("kind")
    @classmethod
    def _kind(cls, v: str) -> str:
        get_kind(v)
        return v

    @model_validator(mode="after")
    def _params(self) -> SignalConfig:
        self.params = check_params(self.kind, self.params)
        return self


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
    ramp_ms: float = Field(
        0.0, ge=0, description="Continuous signals: raised-cosine ramps centred on each event's edges (M7)."
    )


class PoissonSchedule(_M):
    type: Literal["poisson_events"]
    rate_per_hour: float | list[float] = Field(description="Constant, or 24 values by UTC hour.")
    duration: Duration = Field(default_factory=Duration)
    min_gap_s: float = Field(
        0.0,
        ge=0,
        description="One transmitter (M7): an event starting less than this after the previous kept event's end "
        "(within the hour slot) is dropped, so events never overlap. 0: independent events (they may overlap).",
    )
    ramp_ms: float = Field(
        0.0, ge=0, description="Continuous signals: raised-cosine ramps centred on each event's edges (M7)."
    )

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


LEVEL_KEYS = ("p_tx_dbw", "cn0_zenith_dbhz", "cal", "cn0_dbhz", "rx_dbm")


class Level(_M):
    """Satellites: p_tx_dbw, cn0_zenith_dbhz or cal (resolved to p_tx_dbw). Ground and local transmitters (M7): a
    fixed received level, cn0_dbhz or rx_dbm (resolved to rx_dbm)."""

    p_tx_dbw: float | None = Field(None, description="Satellites: power into the transmit antenna.")
    cn0_zenith_dbhz: float | None = Field(
        None, description="Satellites: C/N0 of an overhead pass; solved to p_tx_dbw."
    )
    cal: str | None = Field(
        None, description="Satellites: calibrated level from the profile or band (e.g. meteor_m2_3)."
    )
    cn0_dbhz: float | None = Field(
        None,
        description="Ground/local: received C/N0 (against k T_sys, so the SNR at the emitter's frequency); solved to "
        "rx_dbm.",
    )
    rx_dbm: float | None = Field(None, description="Ground/local: received power at the antenna terminals.")
    extra_loss_db: float = 0.0
    resolved_from: str | None = Field(None, description="Set by resolve: how p_tx_dbw or rx_dbm was obtained.")

    @model_validator(mode="after")
    def _one(self) -> Level:
        given = [k for k in LEVEL_KEYS if getattr(self, k) is not None]
        if self.resolved_from is not None:
            if given not in (["p_tx_dbw"], ["rx_dbm"]):
                raise ValueError("a resolved level holds only p_tx_dbw or rx_dbm")
        elif len(given) != 1:
            raise ValueError(f"give exactly one of {', '.join(LEVEL_KEYS)}")
        return self


class EventLevel(_M):
    """Per-event level in dB relative to the emitter's level: normal(0, sigma_db) clipped to [min_db, max_db], or
    uniform in [min_db, max_db]."""

    dist: Literal["normal", "uniform"] = "normal"
    sigma_db: float = Field(0.0, ge=0)
    min_db: float = 0.0
    max_db: float = 0.0

    @model_validator(mode="after")
    def _range(self) -> EventLevel:
        if self.min_db > self.max_db:
            raise ValueError("min_db must be <= max_db")
        return self


class PerEvent(_M):
    """Draws per event (M7), from the schedule's keyed slot stream: a frequency offset and a level. Used for
    transmitters without geometry, e.g. aircraft VDL bursts (each burst its own offset and power)."""

    offset_hz: float | tuple[float, float] = Field(
        0.0, description="Frequency offset per event (Hz): a number or an inclusive [lo, hi] range (uniform)."
    )
    level_db: EventLevel | None = Field(None, description="Level per event, relative to the emitter's level.")

    @field_validator("offset_hz")
    @classmethod
    def _offset(cls, v: float | tuple[float, float]) -> float | tuple[float, float]:
        if isinstance(v, tuple) and v[0] > v[1]:
            raise ValueError("offset_hz range must be [lo, hi] with lo <= hi")
        return v

    @property
    def max_offset_hz(self) -> float:
        o = self.offset_hz
        return max(abs(o[0]), abs(o[1])) if isinstance(o, tuple) else abs(o)


class Emitter(_M):
    id: str = Field(min_length=1)
    enabled: bool = True
    signal: SignalConfig
    freq_hz: float | list[float] | None = Field(
        None,
        description="Nominal frequency (Hz, quantised to 1 mHz at resolve). Event signals may give a list: each "
        "event is on one of them (a keyed draw per event). Or give `channel`.",
    )
    channel: int | str | None = Field(
        None,
        description="Instead of freq_hz: an index into the source's `channels`, or a channel name from the band "
        "(e.g. orbcomm_gateway). Resolve replaces it by freq_hz.",
    )
    offset_hz: float = Field(0.0, description="Fixed offset (rest-frequency error).")
    oscillator: Oscillator | None = Field(None, description="Per-pass offset and drift (zombies).")
    schedule: ScheduleConfig = Field(default_factory=ContinuousSchedule)
    am: AmConfig | None = None
    level: Level | None = Field(
        None, description="Required, except for wideband signals (noise_comb), whose level is relative to the noise."
    )
    per_event: PerEvent | None = Field(None, description="Frequency offset and level drawn per event (M7).")
    pattern: PatternSpec = Field(default_factory=lambda: PolyPattern(type="poly", coeffs_db=list(QFH_COEFFS_DB)))

    @model_validator(mode="after")
    def _frequency(self) -> Emitter:
        if (self.freq_hz is None) == (self.channel is None):
            raise ValueError(f"emitter {self.id}: give exactly one of freq_hz and channel")
        mode = get_kind(self.signal.kind).mode
        f = self.freq_hz
        if isinstance(f, list):
            if not f:
                raise ValueError(f"emitter {self.id}: freq_hz list is empty")
            if mode != "event":
                raise ValueError(f"emitter {self.id}: a list of frequencies needs an event signal (one per event)")
        if f is not None and min(f if isinstance(f, list) else [f]) <= 0:
            raise ValueError(f"emitter {self.id}: frequencies must be > 0")
        if self.level is None and mode != "wideband":
            raise ValueError(f"emitter {self.id}: level is required")
        if mode == "wideband":
            if self.level is not None or self.per_event is not None or self.am is not None:
                raise ValueError(f"emitter {self.id}: a wideband signal takes no level, per_event or am")
            if not isinstance(self.schedule, PoissonSchedule | ExpGapsSchedule):
                raise ValueError(f"emitter {self.id}: a wideband signal needs an event schedule (episodes)")
        if self.per_event is not None and not isinstance(self.schedule, PoissonSchedule | ExpGapsSchedule):
            raise ValueError(f"emitter {self.id}: per_event needs an event schedule (poisson_events or exp_gaps)")
        return self

    @property
    def lvl(self) -> Level:
        """The level (every non-wideband emitter has one)."""
        if self.level is None:
            raise ValueError(f"emitter {self.id}: no level")
        return self.level

    @property
    def freqs_hz(self) -> list[float]:
        """The emitter's frequencies (one, or the per-event choices). Needs a resolved emitter (no `channel`)."""
        if self.freq_hz is None:
            raise ValueError(f"emitter {self.id}: frequency not resolved (channel {self.channel!r})")
        return list(self.freq_hz) if isinstance(self.freq_hz, list) else [self.freq_hz]


class Source(_M):
    id: str = Field(min_length=1)
    preset: str | None = Field(None, description="Preset the source was expanded from.")
    platform: Literal["satellite", "ground", "local"] = Field(
        "satellite",
        description="satellite (an orbit); ground (M7: a terrestrial transmitter without geometry, fixed received "
        "level); local (M7: interference inside the receiver, e.g. birdies, USB noise).",
    )
    name: str | None = None
    orbit: Orbit | None = Field(None, description="Satellites only (required there).")
    in_catalog: bool | None = Field(None, description="Default: true for catalog orbits, false for synthetic ones.")
    catalog_norad: int | None = Field(None, description="NORAD id of a synthetic object in the catalog output.")
    channels: list[float | str] | None = Field(
        None,
        description="Channel frequencies of this source (Hz, or channel names from the band such as S-5), drawn once "
        "per source in presets (orbcomm: sample([S-1, ..., S-12], 2)). An emitter's `channel: i` uses item i.",
    )
    emitters: list[Emitter] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_emitters(self) -> Source:
        if self.platform == "satellite":
            if self.orbit is None:
                raise ValueError(f"source {self.id}: a satellite needs an orbit")
        else:
            if self.orbit is not None or self.in_catalog or self.catalog_norad is not None:
                raise ValueError(f"source {self.id}: a {self.platform} source has no orbit and is not in the catalog")
            for e in self.emitters:
                if e.oscillator is not None:
                    raise ValueError(f"source {self.id}: emitter {e.id}: per-pass oscillators need a satellite")
        for e in self.emitters:
            if get_kind(e.signal.kind).mode == "wideband" and self.platform != "local":
                raise ValueError(f"source {self.id}: emitter {e.id}: wideband signals belong to a local source")
        ids = [e.id for e in self.emitters]
        if len(ids) != len(set(ids)):
            raise ValueError(f"source {self.id}: duplicate emitter ids {ids}")
        n = len(self.channels or [])
        for e in self.emitters:
            if isinstance(e.channel, int) and not 0 <= e.channel < n:
                raise ValueError(f"source {self.id}: emitter {e.id} uses channel {e.channel}, but channels has {n}")
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
