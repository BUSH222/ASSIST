"""The scene: platforms and emitters built from a resolved scenario (plan §0: a pure function of it)."""

from dataclasses import dataclass, field

from assist.config import schema as S
from assist.config.resolve import pattern_of, t_sys_of
from assist.emitters import link
from assist.emitters.carrier import Carrier, to_mhz
from assist.emitters.emitter import Emitter
from assist.emitters.schedules import (
    AmModel,
    Continuous,
    DurationDist,
    ExpGaps,
    OnOff,
    PoissonEvents,
    Schedule,
)
from assist.platforms.satellite import SatellitePlatform
from assist.rng import KeyedRng
from assist.station import Station
from assist.timebase import parse_iso


@dataclass
class SceneSource:
    id: str
    config: S.Source
    platform: SatellitePlatform
    emitters: list[Emitter] = field(default_factory=list)


def _duration(d: S.Duration) -> DurationDist:
    return DurationDist(d.dist, d.s, d.median_s, d.sigma, d.mean_s, d.max_s)


def make_schedule(sch: S.ScheduleConfig, rng: KeyedRng, sid: str, eid: str) -> Schedule:
    if isinstance(sch, S.ContinuousSchedule):
        return Continuous()
    if isinstance(sch, S.OnOffSchedule):
        if sch.phase_s is None:
            raise ValueError(f"source {sid}: on_off phase not resolved")
        return OnOff(sch.on_s, sch.off_s, sch.phase_s, sch.ramp_ms)
    if isinstance(sch, S.ExpGapsSchedule):
        return ExpGaps(rng, sid, eid, _duration(sch.duration), sch.mean_gap_s, sch.min_gap_s)
    return PoissonEvents(rng, sid, eid, _duration(sch.duration), sch.rate_per_hour)


class Scene:
    def __init__(self, sc: S.Scenario) -> None:
        if sc.clock.start == "now" or sc.assist_version is None:
            raise ValueError("Scene needs a resolved scenario (assist resolve)")
        self.scenario = sc
        self.t_start = parse_iso(sc.clock.start)
        self.t_end = self.t_start + sc.clock.duration
        st = sc.station
        self.station = Station(st.lat, st.lon, st.alt_m, st.tz)
        self.t_sys_k = t_sys_of(sc.frontend)
        self.rng = KeyedRng(sc.seed)
        rx = pattern_of(sc.antenna.pattern)
        horizon = link.HorizonModel()
        self.sources: list[SceneSource] = []
        for src in sc.sources:
            plat = SatellitePlatform(src, self.station, st.horizon_deg)
            ss = SceneSource(src.id, src, plat)
            for e in src.emitters:
                if not e.enabled:
                    continue
                osc = e.oscillator
                carrier = Carrier(
                    src.id,
                    e.id,
                    to_mhz(e.freq_hz) + to_mhz(e.offset_hz),
                    osc.pass_offset_sigma_hz if osc else None,
                    osc.drift_hz_s if osc else None,
                    self.rng,
                )
                am = None
                if e.am is not None:
                    if e.am.phase_rad is None:
                        raise ValueError(f"source {src.id}: AM phase not resolved")
                    am = AmModel(e.am.period_s, e.am.depth_db, e.am.phase_rad)
                if e.level.p_tx_dbw is None or e.level.resolved_from is None:
                    raise ValueError(f"source {src.id}: level not resolved")
                ss.emitters.append(
                    Emitter(
                        e.id,
                        src.id,
                        e.signal.kind,
                        carrier,
                        make_schedule(e.schedule, self.rng, src.id, e.id),
                        am,
                        e.level.p_tx_dbw,
                        pattern_of(e.pattern),
                        e.level.extra_loss_db,
                        plat,
                        rx,
                        self.t_sys_k,
                        horizon,
                    )
                )
            self.sources.append(ss)

    def source(self, source_id: str) -> SceneSource:
        for s in self.sources:
            if s.id == source_id:
                return s
        raise KeyError(source_id)
