"""Per-emitter rendering state (plan §4.1, §4.2): narrow rate and coarse bin, control samples, narrow blocks.

**Control samples** are computed per absolute 1 s chunk at whole Unix milliseconds, so every value depends only on
absolute time, never on where a render starts (window independence, T-WIN):
- amplitude: the link budget (sqrt(mW), no AM or schedule);
- phase, in cycles relative to the coarse bin f_bin = fc + k * 50 Hz (plan §4.1):

      phi(t) = 2 pi * integral(f_tx - f_bin) - 2 pi * f_tx(t) * R(t) / c

  The constant-frequency part of the integral is exact: frequencies are integer mHz and times integer ms, so it is
  reduced modulo one cycle with Python integers. The drift part is (drift / 2) (t - t_aos)^2 from the pass key.
  The range term gives Doppler as exact phase, with no integration drift.

Both are linearly interpolated to the narrow rate. The schedule envelope (with its 5 ms ramps), the AM gain and
the horizon mask are evaluated directly at the narrow rate.

**Narrow rate** (plan §4.2): r >= 1.25 (B + 2 max|f_rx - f_bin|), a multiple of 100 Hz with an FFT-friendly block
length. The excursion bound covers the bin rounding, Doppler at the orbit's perigee speed (plus Earth rotation),
6 sigma of the per-pass offset and the drift over 30 min. Each chunk checks the actual excursion and fails loudly
if it is exceeded.
"""

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.fft as sfft

from assist.emitters.link import C_M_S
from assist.engine.synth import hann
from assist.orbits.geometry import BLOCK_S

if TYPE_CHECKING:
    from assist.emitters.emitter import Emitter
    from assist.orbits.geometry import Pass
    from assist.signals.base import Signal

CONTROL_HZ = 1000
HOPS_PER_CHUNK = 100  # one control chunk = 100 hops = 1 s
RATE_MARGIN = 1.25
MU_KM3_S2 = 398600.8  # WGS72, as SGP4
EARTH_ROTATION_KM_S = 0.6  # bound on what Earth rotation adds to a LEO satellite's speed seen from the ground
OFFSET_SIGMAS = 6.0
DRIFT_BOUND_S = 1800.0
PASS_PAD_S = 0.002  # control samples this close outside a pass use that pass's carrier terms


class RenderError(RuntimeError):
    pass


def max_speed_km_s(sat: object) -> float:
    """Upper bound on a satellite's speed relative to the station: perigee speed plus Earth rotation."""
    model = sat.model  # ty: ignore[unresolved-attribute]
    n = float(model.no_kozai) / 60.0  # rad/s
    e = float(model.ecco)
    a = (MU_KM3_S2 / n**2) ** (1 / 3)
    rp = a * (1 - e)
    return math.sqrt(MU_KM3_S2 * (2 / rp - 1 / a)) + EARTH_ROTATION_KM_S


def narrow_rate(bandwidth_hz: float, excursion_hz: float, preferred: int, fs: int, m: int) -> int:
    """Smallest rate (multiple of 100 Hz, >= preferred) meeting the narrow-rate rule, with a fast FFT length."""
    r = max(int(preferred), math.ceil(RATE_MARGIN * (bandwidth_hz + 2 * excursion_hz) / 100) * 100, 100)
    while True:
        if r > fs:
            raise RenderError(f"narrow rate {r} Hz would exceed the sample rate {fs} Hz")
        n = r * m // fs
        if r * m % fs == 0 and n % 2 == 0 and sfft.next_fast_len(n) == n:
            return r
        r += 100


def _overlaps(intervals: list[tuple[float, float]], a: float, b: float) -> bool:
    return any(lo < b and hi > a for lo, hi in intervals)


def _intersect(xs: list[tuple[float, float]], ys: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out = []
    for a0, a1 in xs:
        for b0, b1 in ys:
            lo, hi = max(a0, b0), min(a1, b1)
            if lo < hi:
                out.append((lo, hi))
    return out


@dataclass(frozen=True)
class Control:
    k0: int  # first control sample, integer Unix ms
    amp: np.ndarray  # link amplitude, sqrt(mW)
    cyc: np.ndarray  # carrier phase in cycles relative to f_bin (a whole number of cycles removed)
    visible: list[tuple[float, float]]  # (aos, los) of the passes overlapping the chunk
    active: list[tuple[float, float]]  # where the emitter may be non-zero: visible and schedule active


class EmitterRender:
    """One emitter as the synthesiser sees it: rate, bin, and its narrow block for any absolute block index."""

    def __init__(self, em: Emitter, signal: Signal, fc_hz: float, fs: int, hop: int, m: int) -> None:
        self.em, self.signal = em, signal
        self.fs, self.hop, self.m = fs, hop, m
        self.hop_ms = hop * 1000 // fs
        self.chunk_ms = HOPS_PER_CHUNK * self.hop_ms
        bin_mhz = fs * 1000 // m  # 50 Hz in mHz
        fc_mhz = round(fc_hz * 1000)
        rest = em.carrier.rest_mhz
        self.k_bin = (rest - fc_mhz + bin_mhz // 2) // bin_mhz
        self.f_bin_mhz = fc_mhz + self.k_bin * bin_mhz
        car = em.carrier
        excursion = (
            abs(rest - self.f_bin_mhz) / 1000.0
            + rest / 1000.0 * max_speed_km_s(em.platform.sat) * 1000.0 / C_M_S
            + OFFSET_SIGMAS * (car.pass_offset_sigma_hz or 0.0)
            + (car.drift_hz_s or 0.0) * DRIFT_BOUND_S
        )
        self.bandwidth_hz = signal.bandwidth_hz()
        self.rate = narrow_rate(self.bandwidth_hz, excursion, signal.preferred_rate(), fs, m)
        self.n = self.rate * m // fs  # narrow samples per block
        self.limit_hz = (self.rate / RATE_MARGIN - self.bandwidth_hz) / 2
        self._ctl: OrderedDict[int, Control] = OrderedDict()
        self._hours: OrderedDict[int, list[Pass]] = OrderedDict()

    @property
    def id(self) -> str:
        return f"{self.em.source_id}/{self.em.id}"

    def _passes(self, t0: float, t1: float) -> list[Pass]:
        """Complete passes overlapping [t0, t1), computed once per UTC hour."""
        found: dict[float, Pass] = {}
        for hr in range(math.floor(t0 / BLOCK_S), math.floor(t1 / BLOCK_S) + 1):
            ps = self._hours.get(hr)
            if ps is None:
                ps = self.em.platform.passes(float(hr * BLOCK_S), float((hr + 1) * BLOCK_S))
                self._hours[hr] = ps
                while len(self._hours) > 4:
                    self._hours.popitem(last=False)
            for p in ps:
                if p.los > t0 and p.aos < t1:
                    found[p.aos] = p
        return [found[k] for k in sorted(found)]

    def control(self, chunk: int) -> Control:
        ctl = self._ctl.get(chunk)
        if ctl is None:
            ctl = self._compute_control(chunk)
            self._ctl[chunk] = ctl
            while len(self._ctl) > 2:
                self._ctl.popitem(last=False)
        return ctl

    def _compute_control(self, chunk: int) -> Control:
        em, car = self.em, self.em.carrier
        k0 = chunk * self.chunk_ms
        k = np.arange(k0, k0 + self.chunk_ms + 2 * self.hop_ms + 1, dtype=np.int64)  # covers the chunk's blocks
        t = k / 1000.0
        t_lo, t_hi = float(t[0]), float(t[-1])
        passes = self._passes(t_lo - 1.0, t_hi + 1.0)
        key = np.full(k.shape, -1, dtype=np.int64)
        base = np.full(k.shape, car.rest_mhz, dtype=np.int64)
        drift = np.zeros(k.shape)
        for p in passes:
            if p.aos_key is None:
                raise RenderError(f"{self.id}: pass without a key")
            sel = (t >= p.aos - PASS_PAD_S) & (t < p.los + PASS_PAD_S)
            off, d = car.pass_terms(p.aos_key)
            key[sel] = p.aos_key
            base[sel] = car.rest_mhz + off
            drift[sel] = d
        g = em.platform.geo.geometry_ms(k)
        amp = 10 ** (em.link_dbm(g.el_deg, g.az_deg, g.range_m, g.off_nadir_deg) / 20)
        # Constant-frequency part: (f - f_bin)[mHz] * t[ms] / 1e6 cycles, reduced mod 1 cycle with integers.
        delta = base - self.f_bin_mhz
        i = k - k0
        cyc = np.empty(k.shape)
        for dv in np.unique(delta):
            sel = delta == dv
            r0 = (int(dv) * k0) % 1_000_000
            cyc[sel] = (r0 + int(dv) * i[sel]) / 1e6
        dt_aos = np.where(key >= 0, (k - 1000 * key) / 1000.0, 0.0)
        f_tx = base / 1000.0 + drift * dt_aos
        cyc += 0.5 * drift * dt_aos**2 - f_tx * g.range_m / C_M_S
        cyc -= math.floor(cyc[0])
        visible = [(p.aos, p.los) for p in passes]
        active = _intersect(visible, em.schedule.active_intervals(t_lo, t_hi + 1e-3))
        self._check_excursion(t, cyc, active)
        return Control(int(k0), amp, cyc, visible, active)

    def _check_excursion(self, t: np.ndarray, cyc: np.ndarray, active: list[tuple[float, float]]) -> None:
        if not active:
            return
        f = np.diff(cyc) * CONTROL_HZ  # Hz relative to f_bin, at the midpoints
        tm = (t[:-1] + t[1:]) / 2
        on = np.zeros(tm.shape, dtype=bool)
        for a, b in active:
            on |= (tm >= a) & (tm < b)
        if on.any():
            worst = float(np.max(np.abs(f[on])))
            if worst > self.limit_hz:
                raise RenderError(
                    f"{self.id}: frequency excursion {worst:.1f} Hz from the bin exceeds {self.limit_hz:.1f} Hz "
                    f"(narrow rate {self.rate} Hz)"
                )

    def block(self, b: int) -> np.ndarray | None:
        """Windowed narrow block v of OLA block b (complex64, n samples), or None when silent throughout."""
        ctl = self.control(b // HOPS_PER_CHUNK)
        tb0 = b * self.hop / self.fs
        tb1 = (b * self.hop + self.m) / self.fs
        if not _overlaps(ctl.active, tb0, tb1):
            return None
        n0 = b * (self.n // 2)
        j = np.arange(self.n)
        t = (n0 + j) / self.rate
        pos = (b * self.hop_ms - ctl.k0) + j * (1000.0 / self.rate)
        grid = np.arange(len(ctl.amp), dtype=np.float64)
        amp = np.interp(pos, grid, ctl.amp)
        cyc = np.interp(pos, grid, ctl.cyc)
        vis = np.zeros(t.shape, dtype=bool)
        for a, z in ctl.visible:
            vis |= (t >= a) & (t < z)
        env = self.em.schedule.envelope(t) * vis
        if self.em.am is not None:
            env = env * 10 ** (self.em.am.gain_db(t) / 20)
        x = self.signal.render(n0, self.n, self.rate)
        v = x * (amp * env * hann(self.n)) * np.exp(2j * np.pi * cyc)
        return v.astype(np.complex64)
