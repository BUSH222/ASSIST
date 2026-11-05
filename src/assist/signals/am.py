"""`am` (plan §5): airband AM voice transmissions (DSB with full carrier), an event signal (M7).

Each event is one press of the push-to-talk: the carrier rises and falls with raised-cosine ramps of `ptt_ramp_ms`
inside the event. The voice is cheap synthetic speech, no words: band-limited noise (audio_lo_hz .. audio_hi_hz, with
a -6 dB/octave tilt above 800 Hz) times a syllable envelope (syllables of 80-250 ms, 1-4 per word, short gaps
inside words, 0.15-0.6 s pauses between words).
- The noise is a continuous stream at the narrow rate, keyed per absolute block of rate / 10 samples (`white` with
  key (source, emitter, "audio")), so any window is rendered on its own; events gate it.
- The syllables of an event come from a keyed stream (source, emitter, "syllables", start in whole microseconds).
- x = (1 + m a(t)) * ramps, with a(t) clipped to [-1, 1]: the carrier has unit power and the voice sidebands add a few
  per cent. The level of an `am` emitter is therefore its carrier's level.
The duration of each event comes from the schedule's duration distribution.
"""

import math
from functools import cache
from typing import TYPE_CHECKING, Any, override

import numpy as np
from pydantic import Field, model_validator
from scipy.signal import firwin2

from assist.signals.base import Signal, SignalParams

if TYPE_CHECKING:
    from types import ModuleType

    from assist.emitters.schedules import Event
    from assist.signals.base import Keys

US = 1_000_000
AUDIO_TAPS = 63
VOICE_SCALE = 2.5  # filtered unit-RMS noise / VOICE_SCALE: peaks near 1 before clipping


class AmParams(SignalParams):
    mod_index: float = Field(0.8, gt=0, le=1)
    ptt_ramp_ms: float = Field(20.0, ge=0)
    audio_lo_hz: float = Field(300.0, gt=0)
    audio_hi_hz: float = Field(3000.0, gt=0)

    @model_validator(mode="after")
    def _check(self) -> AmParams:
        if self.audio_lo_hz >= self.audio_hi_hz:
            raise ValueError("audio_lo_hz must be below audio_hi_hz")
        return self


@cache
def audio_taps(rate: int, lo: float, hi: float) -> np.ndarray:
    """Real FIR (AUDIO_TAPS, float64) for the voice band with a -6 dB/octave tilt above 800 Hz, unit noise gain."""
    nyq = rate / 2
    f = [0.0, 0.8 * lo, lo, max(lo, 800.0), hi, min(nyq, 1.15 * hi), nyq]
    g = [0.0, 0.0, 1.0, 1.0, 800.0 / hi, 0.0, 0.0]
    fr = np.array(f) / nyq
    fr = np.maximum.accumulate(np.clip(fr, 0, 1))
    h = firwin2(AUDIO_TAPS, fr, g)
    h = h / np.sqrt(np.sum(h**2))
    h.flags.writeable = False
    return h


def syllables(g: np.random.Generator, dur: float) -> np.ndarray:
    """(k, 3): start (s from the event start), length (s), amplitude of each syllable in [0, dur)."""
    out = []
    t = float(g.uniform(0.05, 0.3))
    while t < dur:
        for _ in range(int(g.integers(1, 5))):
            ln = float(g.uniform(0.08, 0.25))
            out.append((t, ln, float(g.uniform(0.5, 1.0))))
            t += ln + float(g.uniform(0.02, 0.08))
            if t >= dur:
                break
        t += float(g.uniform(0.15, 0.6))
    return np.array(out, dtype=np.float64).reshape(-1, 3)


class AM(Signal):
    kind = "am"
    Params = AmParams
    mode = "event"

    def __init__(self, params: SignalParams, keys: Keys, dsp: ModuleType) -> None:
        super().__init__(params, keys, dsp)
        self._audio_key = keys.key64("audio")

    @property
    def ap(self) -> AmParams:
        p = self.p
        assert isinstance(p, AmParams)
        return p

    @override
    def bandwidth_hz(self) -> float:
        return 2 * self.ap.audio_hi_hz

    @override
    def preferred_rate(self) -> int:
        return 100

    @override
    @classmethod
    def draw_event(cls, p: SignalParams, g: np.random.Generator) -> tuple[None, dict[str, Any]]:
        return None, {}

    @override
    @classmethod
    def max_duration_s(cls, p: SignalParams) -> None:
        return None

    def audio(self, n0: int, n: int, rate: int) -> np.ndarray:
        """Voice-band noise (float64, unit RMS) at absolute narrow samples [n0, n0 + n)."""
        h = audio_taps(rate, self.ap.audio_lo_hz, self.ap.audio_hi_hz)
        blk = max(1, rate // 10)
        a = n0 - (len(h) - 1)  # history for the filter
        b0, b1 = a // blk, (n0 + n - 1) // blk + 1
        w = self.dsp.white(self._audio_key, b0, b1 - b0, blk).real.astype(np.float64)
        x = w[a - b0 * blk : n0 + n - b0 * blk]
        return np.convolve(x, h, mode="valid")

    def envelope(self, ev: Event, t: np.ndarray) -> np.ndarray:
        """Syllable envelope (0..1) at times t (s from the event start)."""
        g = self.keys.rng.stream(self.keys.source_id, self.keys.emitter_id, "syllables", round(ev.start * US))
        syl = syllables(g, ev.end - ev.start)
        env = np.zeros(t.shape)
        if len(syl) == 0 or t.size == 0:
            return env
        lo, hi = float(t[0]), float(t[-1])
        for s, ln, amp in syl:
            if s + ln < lo or s > hi:
                continue
            m = (t >= s) & (t < s + ln)
            env[m] += amp * np.sin(np.pi * (t[m] - s) / ln) ** 2
        return env

    @override
    def render_events(self, events: list[Event], n0: int, n: int, rate: int) -> np.ndarray:
        p = self.ap
        out = np.zeros(n, dtype=np.complex64)
        r = p.ptt_ramp_ms / 1000
        for ev in events:
            j0 = max(0, math.ceil(ev.start * rate - n0 - 1e-9))
            j1 = min(n, math.floor(ev.end * rate - n0 + 1e-9) + 1)
            if j1 <= j0:
                continue
            t = (n0 + np.arange(j0, j1, dtype=np.int64) - round(ev.start * rate)) / rate  # from the start
            t = np.clip(t, 0.0, None)
            dur = ev.end - ev.start
            a = self.audio(n0 + j0, j1 - j0, rate) / VOICE_SCALE * self.envelope(ev, t)
            x = 1 + p.mod_index * np.clip(a, -1.0, 1.0)
            if r > 0:
                rr = min(r, dur / 2)
                up = t < rr
                x[up] *= 0.5 * (1 - np.cos(np.pi * t[up] / rr))
                dn = t > dur - rr
                x[dn] *= 0.5 * (1 - np.cos(np.pi * np.clip(dur - t[dn], 0, None) / rr))
            out[j0:j1] += x.astype(np.float32)
        return out
