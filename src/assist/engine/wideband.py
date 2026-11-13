"""Wideband local emitters: interference inside the receiver rendered at fs in the time domain (`noise_comb`, the
USB humps). They are added to the signals of a chunk before the ADC: no synthesiser, no receiver response, no
geometry. Levels are relative to the noise density k T_sys (the noise shape's 0 dB).
"""

from typing import TYPE_CHECKING

from assist.emitters.link import K_BOLTZ

if TYPE_CHECKING:
    import numpy as np

    from assist.emitters.emitter import Emitter
    from assist.signals.base import Signal


class WideRender:
    def __init__(self, em: Emitter, signal: Signal, fc_hz: float, fs: int, hop: int, hops: int) -> None:
        self.em, self.signal = em, signal
        self.fs, self.hop, self.hops = fs, hop, hops
        self.n0_w = K_BOLTZ * em.t_sys_k * 1000.0  # mW/Hz
        self.f_offset_hz = em.carrier.rest_mhz / 1000.0 - fc_hz

    @property
    def id(self) -> str:
        return f"{self.em.source_id}/{self.em.id}"

    def chunk(self, c: int) -> np.ndarray | None:
        """Samples of chunk c (complex64, sqrt(mW)), or None when no episode touches it."""
        h0 = c * self.hops
        t0 = h0 * self.hop / self.fs
        evs = self.em.schedule.events(t0, t0 + self.hops * self.hop / self.fs)
        if not evs:
            return None
        return self.signal.render_wide(evs, h0, self.hops, self.hop, self.fs, self.n0_w, self.f_offset_hz)
