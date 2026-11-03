"""The Python reference render engine (plan §4, M3): a scene in, the station's uint8 IQ stream out.

Everything is addressed by absolute sample index n (sample n is at Unix time n / fs, plan §3.1):
- hop h holds samples [hN, hN + N) (N = fs / 100, 10 ms);
- OLA block b covers [bN, bN + 2N), and hop h's output is blocks h-1 and h overlap-added (`synth`);
- per-emitter control is computed per absolute 1 s chunk, noise per absolute hop.
So rendering any window gives exactly the same bytes for the overlapping samples (T-WIN), whatever the start.

Signals pass the receiver response (the profile's noise shape) in the frequency domain; noise is shaped by the same
FIR in time; the ADC maps both to counts (`frontend`). Image, IM3, DC, edge aliasing and the diurnal noise level are
M7; the crystal error (ppm) must be 0 until then.
"""

from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from assist.config.loader import load_band
from assist.engine.control import EmitterRender
from assist.engine.frontend import Frontend, noise_spec
from assist.engine.synth import add_block, overlap, to_time
from assist.signals.base import make_signal

if TYPE_CHECKING:
    from collections.abc import Iterator

    from assist.engine.scene import Scene

CHUNK_HOPS = 100


class Renderer:
    def __init__(self, scene: Scene, base: Path | None = None, noise: bool = True) -> None:
        sc = scene.scenario
        fe = sc.frontend
        if fe.ppm != 0:
            raise ValueError("frontend.ppm must be 0 until M7")
        self.scene = scene
        self.fs = int(fe.fs_hz)
        self.fc_hz = fe.fc_hz
        self.hop = self.fs // 100
        self.m = 2 * self.hop
        base = base or Path.cwd()
        prof = base / sc.profile if sc.profile else None
        self.noise_spec = noise_spec(prof, load_band(sc.band), fe.fc_hz, fe.fs_hz, fe.noise_sigma_counts, sc.profile)
        self.front = Frontend(self.noise_spec, scene.t_sys_k, self.fs, self.hop, self.m, scene.rng)
        self.noise_on = noise
        self.emitters: list[EmitterRender] = []
        for src in scene.sources:
            params = {e.id: e.signal.params for e in src.config.emitters}
            for em in src.emitters:
                sig = make_signal(em.signal_kind, params[em.id])
                self.emitters.append(EmitterRender(em, sig, fe.fc_hz, self.fs, self.hop, self.m))
        self._blocks: OrderedDict[int, np.ndarray | None] = OrderedDict()

    @property
    def gain(self) -> float:
        """ADC gain: counts per sqrt(mW)."""
        return self.front.gain

    def block(self, b: int) -> np.ndarray | None:
        """OLA block b in time (complex64, M samples, sqrt(mW)), or None when no emitter is on."""
        if b in self._blocks:
            return self._blocks[b]
        acc: np.ndarray | None = None
        for er in self.emitters:
            v = er.block(b)
            if v is None:
                continue
            if acc is None:
                acc = np.zeros(self.m, dtype=np.complex64)
            add_block(acc, v, er.k_bin, b)
        y = None if acc is None else to_time(acc, self.front.h_rx)
        self._blocks[b] = y
        while len(self._blocks) > 3:
            self._blocks.popitem(last=False)
        return y

    def hop_iq(self, h: int) -> np.ndarray:
        """Hop h before the ADC: signals plus noise (complex64, sqrt(mW))."""
        y = overlap(self.block(h - 1), self.block(h), self.hop)
        if self.noise_on:
            y += self.front.noise(h)
        return y

    def _hops(self, n0: int, n: int) -> Iterator[tuple[int, int, int]]:
        """(hop, first, last) sample offsets within each hop covering absolute samples [n0, n0 + n)."""
        if n <= 0:
            return
        for h in range(n0 // self.hop, (n0 + n - 1) // self.hop + 1):
            lo = max(n0, h * self.hop) - h * self.hop
            hi = min(n0 + n, (h + 1) * self.hop) - h * self.hop
            yield h, lo, hi

    def iq(self, n0: int, n: int) -> np.ndarray:
        """Absolute samples [n0, n0 + n) before the ADC (complex64, sqrt(mW))."""
        parts = [self.hop_iq(h)[lo:hi] for h, lo, hi in self._hops(n0, n)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.complex64)

    def counts(self, n0: int, n: int) -> np.ndarray:
        """As iq, scaled to ADC counts around 127.5 (not quantised or clipped)."""
        return self.iq(n0, n) * np.float32(self.gain)

    def iter_u8(self, n0: int, n: int, chunk_hops: int = CHUNK_HOPS) -> Iterator[np.ndarray]:
        """The uint8 stream (interleaved I, Q) for absolute samples [n0, n0 + n), in pieces of about chunk_hops."""
        parts: list[np.ndarray] = []
        for h, lo, hi in self._hops(n0, n):
            parts.append(self.front.adc(self.hop_iq(h)[lo:hi]))
            if len(parts) >= chunk_hops:
                yield np.concatenate(parts)
                parts = []
        if parts:
            yield np.concatenate(parts)

    def u8(self, n0: int, n: int) -> np.ndarray:
        parts = list(self.iter_u8(n0, n))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint8)
