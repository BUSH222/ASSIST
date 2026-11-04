"""Noise-like signals (plan §5, M7): `noise_band` and `noise_comb`.

`noise_band` (continuous): complex Gaussian noise with a Gaussian spectrum of full width at half maximum `width_hz`,
unit power. White noise at the narrow rate is keyed per absolute block of rate / 10 samples (`white` with key
(source, emitter, "noise_band")) and filtered; the filter's history comes from the previous block, so any window is
rendered on its own. Used for wide birdies.

`noise_comb` (wideband, local): the USB humps, a comb of noise ripples across the whole band that wanders. Each
episode (an event of the emitter's schedule) adds h(t) = g (w(t) + e^(j theta) w(t - D)) to the stream before the ADC,
with w complex white noise at fs: its spectrum is 2 g^2 (2 + 2 cos(2 pi f D / fs - theta)), a comb of humps
fs / D apart whose teeth sit at theta / 2 pi of the spacing. g sets the ripple: the noise spectrum becomes
N (1 + beta (1 + cos)), whose ripple amplitude (dB, peak) is A = 10 log10(1 + 2 beta) / 2 in the passband, where the
noise density N is k T_sys. theta follows the comb's position c(t) = c0 + integral of v, with v piecewise linear
between normal(0, wander_hz_s) knots every knot_s seconds (a keyed stream per episode), evaluated per hop. Raised-
cosine ramps of ramp_s start and end each episode. Per episode (drawn from the slot stream): the spacing (one of
`spacing_hz`), the ripple amplitude (`amplitude_db`, a number or [lo, hi]) and c0. The white noise is keyed per
episode and absolute hop.
"""

import math
from functools import cache
from typing import TYPE_CHECKING, Any, Literal, override

import numpy as np
from pydantic import Field, field_validator

from assist.signals.base import Signal, SignalParams, float_range

if TYPE_CHECKING:
    from types import ModuleType

    from assist.emitters.schedules import Event
    from assist.signals.base import Keys

US = 1_000_000
FWHM_TO_SIGMA = 1 / (2 * math.sqrt(2 * math.log(2)))


class NoiseBandParams(SignalParams):
    width_hz: float = Field(gt=0, description="Full width at half maximum of the spectrum (Hz).")
    shape: Literal["gaussian"] = "gaussian"


@cache
def gauss_taps(width_hz: float, rate: int) -> np.ndarray:
    """Real Gaussian FIR (float64) whose power response |H|^2 has FWHM width_hz; sum of squares 1/2 (complex
    unit-variance-per-component white noise in, unit power out)."""
    sigma_amp = math.sqrt(2) * width_hz * FWHM_TO_SIGMA  # |H| has sqrt(2) times the sigma of |H|^2
    sigma_t = 1 / (2 * math.pi * sigma_amp)  # s
    k = max(1, math.ceil(4 * sigma_t * rate))
    t = np.arange(-k, k + 1, dtype=np.float64) / rate
    h = np.exp(-0.5 * (t / sigma_t) ** 2)
    h = h / np.sqrt(2 * np.sum(h**2))
    h.flags.writeable = False
    return h


class NoiseBand(Signal):
    kind = "noise_band"
    Params = NoiseBandParams

    def __init__(self, params: SignalParams, keys: Keys, dsp: ModuleType) -> None:
        super().__init__(params, keys, dsp)
        self._key = keys.key64("noise_band")

    @property
    def np_(self) -> NoiseBandParams:
        p = self.p
        assert isinstance(p, NoiseBandParams)
        return p

    @override
    def bandwidth_hz(self) -> float:
        return 2 * self.np_.width_hz

    @override
    def preferred_rate(self) -> int:
        return 100

    @override
    def render(self, n0: int, n: int, rate: int) -> np.ndarray:
        h = gauss_taps(self.np_.width_hz, rate)
        blk = max(1, rate // 10)
        a = n0 - (len(h) - 1)
        b0, b1 = a // blk, (n0 + n - 1) // blk + 1
        w = self.dsp.white(self._key, b0, b1 - b0, blk)
        x = w[a - b0 * blk : n0 + n - b0 * blk]
        y = np.convolve(x.real.astype(np.float64), h, mode="valid") + 1j * np.convolve(
            x.imag.astype(np.float64), h, mode="valid"
        )
        return y.astype(np.complex64)


class NoiseCombParams(SignalParams):
    spacing_hz: float | list[float] = Field(
        [80_000.0, 141_176.0, 150_000.0],
        description="Tooth spacing (Hz), or a list to choose from per episode; the teeth are fs / round(fs / "
        "spacing) apart.",
    )
    amplitude_db: float | tuple[float, float] = Field(
        (0.1, 0.3), description="Ripple amplitude (dB, peak) per episode: a number or an inclusive [lo, hi] range."
    )
    wander_hz_s: float = Field(1500.0, ge=0, description="RMS speed of the comb's drift (Hz/s).")
    knot_s: float = Field(20.0, gt=0, description="Spacing of the drift-speed knots (s).")
    ramp_s: float = Field(2.0, ge=0, description="Raised-cosine ramps at the start and end of an episode (s).")

    @field_validator("spacing_hz")
    @classmethod
    def _spacing(cls, v: float | list[float]) -> float | list[float]:
        vals = v if isinstance(v, list) else [v]
        if not vals or min(vals) <= 0:
            raise ValueError("spacing_hz must be > 0 (a number or a non-empty list)")
        return v

    @field_validator("amplitude_db")
    @classmethod
    def _amp(cls, v: float | tuple[float, float]) -> float | tuple[float, float]:
        lo, hi = float_range(v)
        if not 0 < lo <= hi:
            raise ValueError("amplitude_db must be > 0 (lo <= hi)")
        return v


def beta_of(amplitude_db: float) -> float:
    """Comb power (relative to the noise density) for a ripple amplitude A (dB, peak): 1 + 2 beta = 10^(2A/10)."""
    return float((10 ** (2 * amplitude_db / 10) - 1) / 2)


class NoiseComb(Signal):
    kind = "noise_comb"
    Params = NoiseCombParams
    mode = "wideband"

    @property
    def cp(self) -> NoiseCombParams:
        p = self.p
        assert isinstance(p, NoiseCombParams)
        return p

    @override
    def bandwidth_hz(self) -> float:
        return 0.0

    @override
    def preferred_rate(self) -> int:
        return 100

    @override
    @classmethod
    def draw_event(cls, p: SignalParams, g: np.random.Generator) -> tuple[None, dict[str, Any]]:
        assert isinstance(p, NoiseCombParams)
        sp = p.spacing_hz if isinstance(p.spacing_hz, list) else [p.spacing_hz]
        spacing = sp[int(g.integers(len(sp)))] if len(sp) > 1 else sp[0]
        lo, hi = float_range(p.amplitude_db)
        amp = float(g.uniform(lo, hi)) if hi > lo else lo
        return None, {"spacing_hz": float(spacing), "amplitude_db": amp, "phase": float(g.uniform(0.0, 1.0))}

    @override
    @classmethod
    def max_duration_s(cls, p: SignalParams) -> None:
        return None

    def position_hz(self, ev: Event, t: np.ndarray, spacing: float) -> np.ndarray:
        """Comb position c(t) (Hz, unwrapped) at times t (s from the episode start)."""
        p = self.cp
        dur = ev.end - ev.start
        nk = math.ceil(dur / p.knot_s) + 2
        g = self.keys.rng.stream(self.keys.source_id, self.keys.emitter_id, "wander", round(ev.start * US))
        v = g.normal(0.0, p.wander_hz_s, nk)
        k = p.knot_s
        cum = np.concatenate([[0.0], np.cumsum(k * (v[:-1] + v[1:]) / 2)])
        i = np.clip(np.floor(t / k).astype(np.int64), 0, nk - 2)
        tau = t - i * k
        return float(ev.params["phase"]) * spacing + cum[i] + v[i] * tau + (v[i + 1] - v[i]) * tau**2 / (2 * k)

    @override
    def render_wide(
        self, events: list[Event], h0: int, nhops: int, hop: int, fs: int, n0_w: float, f_offset_hz: float = 0.0
    ) -> np.ndarray:
        out = np.zeros(nhops * hop, dtype=np.complex64)
        r = self.cp.ramp_s
        for ev in events:
            ha = max(h0, math.floor(ev.start * fs / hop))
            hb = min(h0 + nhops, math.ceil(ev.end * fs / hop))
            if hb <= ha:
                continue
            d = max(1, round(fs / float(ev.params["spacing_hz"])))
            spacing = fs / d
            w = self.dsp.white(self.keys.key64("comb", round(ev.start * US)), ha - 1, hb - ha + 1, hop)
            x = w[hop:].reshape(-1, hop)
            xd = w[hop - d : len(w) - d].reshape(-1, hop)
            t = ((np.arange(ha, hb, dtype=np.int64) * hop + hop // 2) / fs) - ev.start  # hop centres
            theta = 2 * np.pi * ((self.position_hz(ev, t, spacing) + f_offset_hz) / spacing % 1.0)
            dur = ev.end - ev.start
            env = np.ones(t.shape)
            if r > 0:
                rr = min(r, dur / 2)
                env *= 0.5 * (1 - np.cos(np.pi * np.clip(t / rr, 0, 1)))
                env *= 0.5 * (1 - np.cos(np.pi * np.clip((dur - t) / rr, 0, 1)))
            gh = math.sqrt(beta_of(float(ev.params["amplitude_db"])) * n0_w * fs / 4) * env
            seg = out[(ha - h0) * hop : (hb - h0) * hop].reshape(-1, hop)
            seg += xd * (gh * np.exp(1j * theta)).astype(np.complex64)[:, None]  # complex64 throughout
            seg += x * gh.astype(np.float32)[:, None]
        return out
