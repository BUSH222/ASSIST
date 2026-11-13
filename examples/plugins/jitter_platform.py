"""`tethered`: a transmitter in a fixed direction whose distance swings, e.g. a beacon on a tethered balloon.

An external ASSIST platform: this file is everything it needs. A scenario uses it by listing the file:

    plugins:
      - path: examples/plugins/jitter_platform.py
    sources:
      - id: balloon
        platform: tethered
        platform_params: {az_deg: 200, el_deg: 35, range_m: 3000, swing_m: 50, period_ms: 5000}
        emitters:
          - id: beacon
            signal: {kind: cw}
            freq_hz: 137.3e6
            level: {cn0_dbhz: 50}

What a platform does: it says where the source is at any absolute time (`geometry`) and when it is visible
(`passes`). The engine turns the range into carrier phase (so the swing shows as Doppler), and the truth output lists
the direction, range and range rate. The emitter's level is a received level (`cn0_dbhz` or `rx_dbm`), as for a
`ground` source: orbits and the link budget belong to the core `satellite` platform.

Each source's direction is offset by a small jitter and its swing has its own phase. Both are drawn once from the
platform's RNG scope, whose keys are ("platform", "tethered", source id, ...) under the scenario seed: the same
scenario always gives the same values, and they do not depend on which other sources or modules exist.

The geometry is a pure function of the time, and exact at the renderer's millisecond grid (the swing phase is
formed from integer milliseconds), so any window of a render gives the same samples.
"""

import math
from typing import override

import numpy as np
from pydantic import Field

from assist import api

__version__ = "0.1.0"
ASSIST_API = "1.0"  # the assist.api version this module was written for

DAY_S = 86400


class TetheredParams(api.PlatformParams):
    az_deg: float = Field(180.0, ge=0, lt=360, description="Azimuth of the mean direction.")
    el_deg: float = Field(30.0, gt=0, le=90, description="Elevation of the mean direction.")
    range_m: float = Field(2000.0, gt=0, description="Mean distance from the station.")
    swing_m: float = Field(50.0, ge=0, description="Amplitude of the distance swing (must be below range_m).")
    period_ms: int = Field(5000, gt=0, description="Period of the swing (whole milliseconds).")
    jitter_deg: float = Field(2.0, ge=0, description="Largest per-source offset of azimuth and of elevation.")


@api.register_platform("tethered")
class Tethered(api.Platform):
    Params = TetheredParams
    # Capabilities are the defaults of api.Platform: it has geometry (fixed = False), takes a received level, has no
    # orbit or catalog entry, no per-pass oscillator, no fade at the horizon, and carries no wideband signals.

    def __init__(self, ctx: api.PlatformContext) -> None:
        p = ctx.params
        assert isinstance(p, TetheredParams)
        if p.swing_m >= p.range_m:
            raise ValueError(f"source {ctx.source_id}: swing_m must be below range_m")
        self.source_id, self.name = ctx.source_id, ctx.name
        self.p = p
        g = ctx.scope.stream("jitter")  # one stream per source: direction offsets first, then the swing phase
        self.az_deg = (p.az_deg + float(g.uniform(-p.jitter_deg, p.jitter_deg))) % 360.0
        self.el_deg = min(90.0, max(0.1, p.el_deg + float(g.uniform(-p.jitter_deg, p.jitter_deg))))
        self.phase = float(g.uniform(0.0, 1.0))  # of the swing, in periods
        self.omega = 2 * math.pi * 1000.0 / p.period_ms  # rad/s

    @override
    @classmethod
    def build(cls, ctx: api.PlatformContext) -> Tethered:
        return cls(ctx)

    def _at(self, periods: np.ndarray) -> api.Geometry:
        """Geometry at a swing phase given in periods (any whole number of periods may be dropped)."""
        x = 2 * np.pi * (periods + self.phase)
        full = np.ones(periods.shape)
        return api.Geometry(
            el_deg=self.el_deg * full,
            az_deg=self.az_deg * full,
            range_m=self.p.range_m + self.p.swing_m * np.sin(x),
            range_rate_m_s=self.p.swing_m * self.omega * np.cos(x),
            off_nadir_deg=0.0 * full,
        )

    @override
    def geometry(self, t_unix: np.ndarray) -> api.Geometry:
        t = np.atleast_1d(np.asarray(t_unix, dtype=np.float64))
        return self._at(np.mod(t * 1000.0, self.p.period_ms) / self.p.period_ms)

    @override
    def geometry_ms(self, k_ms: np.ndarray) -> api.Geometry:
        k = np.asarray(k_ms, dtype=np.int64)
        return self._at((k % self.p.period_ms) / self.p.period_ms)  # exact: the phase comes from integers

    @override
    def passes(self, t0: float, t1: float) -> list[api.Pass]:
        """Always visible: one "pass" per UTC day, so a pass and its key never depend on the window asked for."""
        days = range(math.floor(t0 / DAY_S), math.floor(max(t0, np.nextafter(t1, t0)) / DAY_S) + 1)
        return [
            api.Pass(0, self.name, float(d * DAY_S), (d + 0.5) * DAY_S, float((d + 1) * DAY_S), self.el_deg, d * DAY_S)
            for d in days
        ]

    @override
    def max_speed_km_s(self) -> float:
        return self.p.swing_m * self.omega / 1000.0
