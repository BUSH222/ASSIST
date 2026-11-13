"""The scene: platforms and emitters built from a resolved scenario (a pure function of it)."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from assist import plugins
from assist.config import schema as S
from assist.config.resolve import pattern_of, t_sys_of
from assist.emitters import link
from assist.emitters.carrier import Carrier, to_mhz
from assist.emitters.emitter import Emitter
from assist.emitters.schedules import (
    AmModel,
    Continuous,
    DurationDist,
    EventMaker,
    ExpGaps,
    OnOff,
    PoissonEvents,
    Schedule,
)
from assist.platforms.base import Platform, PlatformContext, platform_class
from assist.rng import KeyedRng
from assist.signals.base import event_maker, float_range, get_kind
from assist.station import Station
from assist.timebase import parse_iso

if TYPE_CHECKING:
    import numpy as np

    from assist.rng import RngScope


@dataclass
class SceneSource:
    id: str
    config: S.Source
    platform: Platform
    emitters: list[Emitter] = field(default_factory=list)


def _duration(d: S.Duration) -> DurationDist:
    return DurationDist(d.dist, d.s, d.median_s, d.sigma, d.mean_s, d.max_s)


def _draw_per_event(pe: S.PerEvent, g: np.random.Generator) -> dict[str, Any]:
    """Per-event frequency offset (rounded to 1 mHz) and level (dB), from the slot stream."""
    out: dict[str, Any] = {}
    lo, hi = float_range(pe.offset_hz)
    off = float(g.uniform(lo, hi)) if hi > lo else lo
    out["offset_hz"] = round(off * 1000) / 1000
    lv = pe.level_db
    if lv is not None:
        if lv.dist == "uniform":
            x = float(g.uniform(lv.min_db, lv.max_db)) if lv.max_db > lv.min_db else lv.min_db
        else:
            x = min(max(float(g.normal(0.0, lv.sigma_db)), lv.min_db), lv.max_db)
        out["level_db"] = x
    return out


def make_maker(e: S.Emitter) -> EventMaker | None:
    """Event and wideband signals, and emitters with per-event draws: each event's duration and parameters, drawn in
    this order from the slot stream: the frequency index of a multi-frequency emitter (only when there is a
    choice), the signal's parameters and duration (or the schedule's duration distribution), the per-event offset
    and level."""
    mode = get_kind(e.signal.kind).mode
    pe = e.per_event
    if mode == "continuous" and pe is None:
        return None
    cls, p = event_maker(e.signal.kind, e.signal.params) if mode != "continuous" else (None, None)
    sch = e.schedule
    if not isinstance(sch, S.ExpGapsSchedule | S.PoissonSchedule):
        raise ValueError(f"emitter {e.id} has an event signal, so it needs exp_gaps or poisson_events")
    dur = _duration(sch.duration)
    freqs = e.freqs_hz

    def draw(g: np.random.Generator) -> tuple[float, dict[str, Any]]:
        extra: dict[str, Any] = {}
        if len(freqs) > 1:
            ch = int(g.integers(len(freqs)))
            extra = {"channel": ch, "freq_hz": freqs[ch]}
        d: float | None = None
        params: dict[str, Any] = {}
        if cls is not None and p is not None:
            d, params = cls.draw_event(p, g)
        if d is None:
            d = float(dur.draw(g, 1)[0])
            params = {"duration_s": d, **params}
        if pe is not None:
            extra.update(_draw_per_event(pe, g))
        return d, {**params, **extra}

    upper = cls.max_duration_s(p) if cls is not None and p is not None else None
    return EventMaker(draw, dur.upper_s if upper is None else upper)


def make_schedule(
    sch: S.ScheduleConfig, scope: RngScope, sid: str, eid: str, maker: EventMaker | None = None
) -> Schedule:
    if maker is not None and not isinstance(sch, S.ExpGapsSchedule | S.PoissonSchedule):
        raise ValueError(f"source {sid}: emitter {eid} has an event signal, so it needs exp_gaps or poisson_events")
    if isinstance(sch, S.ContinuousSchedule):
        return Continuous()
    if isinstance(sch, S.OnOffSchedule):
        if sch.phase_s is None:
            raise ValueError(f"source {sid}: on_off phase not resolved")
        return OnOff(sch.on_s, sch.off_s, sch.phase_s, sch.ramp_ms)
    if isinstance(sch, S.ExpGapsSchedule):
        return ExpGaps(scope, _duration(sch.duration), sch.mean_gap_s, sch.min_gap_s, maker, sch.ramp_ms)
    return PoissonEvents(scope, _duration(sch.duration), sch.rate_per_hour, maker, sch.ramp_ms, sch.min_gap_s)


class Scene:
    def __init__(self, sc: S.Scenario) -> None:
        if sc.clock.start == "now" or sc.assist_version is None:
            raise ValueError("Scene needs a resolved scenario (assist resolve)")
        # The scenario's external modules (and no others) must be the active ones while it is built and rendered.
        plugins.activate([p.model_dump(exclude_none=True) for p in sc.plugins], (Path.cwd(),))
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
            pcls = platform_class(src.platform)
            plat = pcls.build(
                PlatformContext(
                    src.id,
                    src.name or src.id,
                    pcls.Params.model_validate(src.platform_params),
                    self.station,
                    st.horizon_deg,
                    self.rng.scope("platform", src.platform, src.id),
                    src,
                )
            )
            ss = SceneSource(src.id, src, plat)
            for e in src.emitters:
                if not e.enabled:
                    continue
                osc = e.oscillator
                freqs = tuple(to_mhz(f) + to_mhz(e.offset_hz) for f in e.freqs_hz)
                scope = self.rng.scope(src.id, e.id)
                carrier = Carrier(
                    freqs[0],
                    osc.pass_offset_sigma_hz if osc else None,
                    osc.drift_hz_s if osc else None,
                    scope,
                )
                am = None
                if e.am is not None:
                    if e.am.phase_rad is None:
                        raise ValueError(f"source {src.id}: AM phase not resolved")
                    am = AmModel(e.am.period_s, e.am.depth_db, e.am.phase_rad)
                lv = e.level
                if lv is not None and (lv.resolved_from is None or (lv.p_tx_dbw is None and lv.rx_dbm is None)):
                    raise ValueError(f"source {src.id}: level not resolved")
                ss.emitters.append(
                    Emitter(
                        e.id,
                        src.id,
                        e.signal.kind,
                        carrier,
                        make_schedule(e.schedule, scope, src.id, e.id, make_maker(e)),
                        am,
                        math.nan if lv is None or lv.p_tx_dbw is None else lv.p_tx_dbw,
                        pattern_of(e.pattern),
                        0.0 if lv is None else lv.extra_loss_db,
                        plat,
                        rx,
                        self.t_sys_k,
                        horizon,
                        freqs if len(freqs) > 1 else (),
                        None if lv is None else lv.rx_dbm,
                        0.0 if e.per_event is None else e.per_event.max_offset_hz,
                        e.per_event is not None,
                        st.fade_s if pcls.fades_at_horizon else 0.0,
                    )
                )
            self.sources.append(ss)

    def source(self, source_id: str) -> SceneSource:
        for s in self.sources:
            if s.id == source_id:
                return s
        raise KeyError(source_id)
