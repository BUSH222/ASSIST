"""Emitter (plan §3.4): signal + carrier + schedule + level + pattern on a platform, and its truth.

Truth values follow the plan's definitions:
- f_rest = f_nominal + offset_hz; f_tx adds the per-pass offset and drift (§6.3). An event emitter with several
  frequencies (LoRa channels) has one f_rest per frequency; each event's parameters say which (`channel`).
- doppler = -f_tx * range_rate / c (first order, geometry at reception time, §4.1); f_rx = f_tx + doppler.
- rx_dbm from the link budget (§6.2) with the AM gain at that instant; cn0 = rx_dbm - 30 - 10 log10(k T_sys).
- active_fraction: share of [t, t + step) during which the schedule is on (half-amplitude edges).
Frequencies are carried in integer mHz (see carrier.py), so the printed truth is exact to 1 mHz.
"""

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import numpy as np

from assist.emitters import link
from assist.orbits.geometry import BLOCK_S

if TYPE_CHECKING:
    from assist.emitters.carrier import Carrier
    from assist.emitters.schedules import AmModel, Event, Schedule
    from assist.platforms.satellite import SatellitePlatform


@dataclass
class TruthArrays:
    t: np.ndarray
    f_rest_mhz: int
    f_tx_mhz: np.ndarray  # integer mHz
    doppler_mhz: np.ndarray  # integer mHz; f_rx = f_tx + doppler exactly in mHz
    el_deg: np.ndarray
    az_deg: np.ndarray
    range_m: np.ndarray
    range_rate_m_s: np.ndarray
    rx_dbm: np.ndarray
    cn0_dbhz: np.ndarray
    active_fraction: np.ndarray
    f_tx_hz: np.ndarray  # float versions for computation
    doppler_hz: np.ndarray

    @property
    def f_rx_mhz(self) -> np.ndarray:
        return self.f_tx_mhz + self.doppler_mhz

    @property
    def f_rx_hz(self) -> np.ndarray:
        return self.f_tx_hz + self.doppler_hz


@dataclass
class EventTruth:
    event: Event
    f_rx_mhz: int
    rx_dbm: float


@dataclass
class Emitter:
    id: str
    source_id: str
    signal_kind: str
    carrier: Carrier
    schedule: Schedule
    am: AmModel | None
    p_tx_dbw: float
    tx_pattern: link.Pattern
    extra_loss_db: float
    platform: SatellitePlatform
    rx_pattern: link.Pattern
    t_sys_k: float
    horizon: link.HorizonModel
    freqs_mhz: tuple[int, ...] = ()  # rest frequency per channel (f_nominal + offset); () = just carrier.rest_mhz

    @property
    def is_event(self) -> bool:
        return self.schedule.is_event

    @property
    def channels_mhz(self) -> tuple[int, ...]:
        return self.freqs_mhz or (self.carrier.rest_mhz,)

    def event_channel(self, ev: Event) -> int:
        return int(ev.params.get("channel", 0))

    def link_dbm(
        self, el: np.ndarray, az: np.ndarray, rng_m: np.ndarray, off_nadir: np.ndarray, f_mhz: int | None = None
    ) -> np.ndarray:
        """Received level (dBm) from the link budget alone: no AM, no schedule (plan §6.2); at f_mhz if given."""
        return link.p_rx_dbm(
            self.p_tx_dbw,
            self.tx_pattern.gain_db(off_nadir),
            rng_m,
            (self.carrier.rest_mhz if f_mhz is None else f_mhz) / 1000.0,
            self.rx_pattern.gain_db(90.0 - el),
            self.horizon.loss_db(az, el),
            self.extra_loss_db,
        )

    def level(
        self,
        t: np.ndarray,
        el: np.ndarray,
        az: np.ndarray,
        rng_m: np.ndarray,
        off_nadir: np.ndarray,
        f_mhz: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(rx_dbm, cn0_dbhz) at the given geometry, including AM."""
        rx = self.link_dbm(el, az, rng_m, off_nadir, f_mhz)
        if self.am is not None:
            rx = rx + self.am.gain_db(t)
        return rx, rx - 30 - link.kt_dbw_hz(self.t_sys_k)

    def truth(self, t_unix: np.ndarray, step_s: float) -> TruthArrays:
        """Truth at the times where the platform is above the horizon (other times are dropped).

        Long time spans are processed one geometry block (hour) at a time, so the block cache stays small.
        """
        t_all = np.asarray(t_unix, dtype=np.float64)
        hours = np.floor(t_all / BLOCK_S)
        parts = [self._truth_chunk(t_all[hours == h], step_s) for h in np.unique(hours)]
        if len(parts) == 1:
            return parts[0]
        if not parts:
            return self._truth_chunk(t_all, step_s)
        arrays = {
            f.name: np.concatenate([getattr(p, f.name) for p in parts])
            for f in fields(TruthArrays)
            if f.name != "f_rest_mhz"
        }
        return TruthArrays(f_rest_mhz=self.carrier.rest_mhz, **arrays)

    def _truth_chunk(self, t_all: np.ndarray, step_s: float) -> TruthArrays:
        passes = self.platform.passes(float(t_all.min()), float(t_all.max()) + step_s) if t_all.size else []
        keys_all = self.platform.geo.pass_keys(t_all, passes)
        vis = keys_all >= 0
        t, keys = t_all[vis], keys_all[vis]
        g = self.platform.geometry(t)
        base, small = self.carrier.f_tx(t, keys)
        f_tx_mhz = base + np.round(small * 1000).astype(np.int64)
        f_tx_hz = base / 1000.0 + small
        doppler_hz = -f_tx_hz * g.range_rate_m_s / link.C_M_S
        rx, cn0 = self.level(t, g.el_deg, g.az_deg, g.range_m, g.off_nadir_deg)
        return TruthArrays(
            t,
            self.carrier.rest_mhz,
            f_tx_mhz,
            np.round(doppler_hz * 1000).astype(np.int64),
            g.el_deg,
            g.az_deg,
            g.range_m,
            g.range_rate_m_s,
            rx,
            cn0,
            self.schedule.active_fraction(t, step_s),
            f_tx_hz,
            doppler_hz,
        )

    def event_truth(self, t0: float, t1: float) -> list[EventTruth]:
        """Events overlapping [t0, t1) whose midpoint is above the horizon; frequency and level at the midpoint
        (each on its own channel)."""
        events = self.schedule.events(t0, t1)
        if not events:
            return []
        mid = np.array([(e.start + e.end) / 2 for e in events])
        tr = self.truth(mid, 1.0)
        at = {float(x): i for i, x in enumerate(tr.t)}
        chans = self.channels_mhz
        out = []
        for e, m in zip(events, mid, strict=True):
            i = at.get(float(m))
            if i is None:
                continue
            shift = chans[self.event_channel(e)] - self.carrier.rest_mhz  # mHz
            if shift == 0:
                out.append(EventTruth(e, int(tr.f_rx_mhz[i]), float(tr.rx_dbm[i])))
                continue
            f_tx_hz = float(tr.f_tx_hz[i]) + shift / 1000.0
            doppler_mhz = round(-f_tx_hz * float(tr.range_rate_m_s[i]) / link.C_M_S * 1000)
            g = self.platform.geometry(np.array([m]))
            rx, _ = self.level(
                np.array([m]), g.el_deg, g.az_deg, g.range_m, g.off_nadir_deg, chans[self.event_channel(e)]
            )
            out.append(EventTruth(e, int(tr.f_tx_mhz[i]) + shift + doppler_mhz, float(rx[0])))
        return out
