"""The render engine (plan §4): a scene in, the station's uint8 IQ stream out.

Everything is addressed by absolute sample index n (sample n is at Unix time n / fs, plan §3.1):
- hop h holds samples [hN, hN + N) (N = fs / 100, 10 ms);
- OLA block b covers [bN, bN + 2N), and hop h's output is blocks h-1 and h overlap-added (`synth`);
- the engine works in absolute 1 s chunks: chunk c is hops 100c .. 100c + 99, i.e. samples [c fs, (c + 1) fs). Per
  emitter, control samples and narrow samples are computed per chunk; noise is drawn per hop.
So rendering any window gives exactly the same bytes for the overlapping samples (T-WIN), whatever the start.

Per chunk there is one kernel call each for the signal blocks (`dsp.synth`), the shaped noise (`dsp.noise`) and the
ADC (`dsp.adc`), from `assist.backend` (C++, or the numpy twins with ASSIST_PURE=1). They write into buffers the
Renderer keeps (fresh 19 MB arrays every chunk cost page faults, which contend between worker processes), so the
per-chunk results (`signal`, `chunk_iq`, `chunk_u8`) are views valid until the next chunk replaces them.

Signals pass the receiver response (the profile's noise shape) in the frequency domain; noise is shaped by the same
FIR in time. Wideband local emitters (M7: the USB humps) are added in the time domain before the ADC, without the
receiver response. Image, IM3, DC, edge aliasing and the diurnal noise level are M8; the crystal error (ppm) must
be 0 until then.
"""

import math
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from assist.config.loader import load_band
from assist.engine.control import HOPS_PER_CHUNK, EmitterRender
from assist.engine.frontend import Frontend, noise_spec
from assist.engine.synth import synth_blocks
from assist.engine.wideband import WideRender
from assist.orbits.geometry import BLOCK_S
from assist.signals.base import Keys, make_signal

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

    from assist.engine.control import Stream
    from assist.engine.scene import Scene

CHUNK_HOPS = HOPS_PER_CHUNK  # 1 s
PREFETCH_S = 60.0  # live serving: next hour's passes and geometry are computed during the last minute of an hour
STAGES = ("control", "synth", "wideband", "noise", "adc", "prefetch")


class Renderer:
    """Renders a scene by absolute sample index. `dsp`: kernel backend (default assist.backend.dsp).

    `prefetch` (for `serve`): during the last minute of each UTC hour, compute the next hour's passes and geometry
    a few emitters per chunk, so the hour boundary costs no more than any other second.
    """

    def __init__(
        self,
        scene: Scene,
        base: Path | None = None,
        noise: bool = True,
        dsp: ModuleType | None = None,
        prefetch: bool = False,
    ) -> None:
        sc = scene.scenario
        fe = sc.frontend
        if fe.ppm != 0:
            raise ValueError("frontend.ppm must be 0 until M8")
        if dsp is None:
            from assist.backend import dsp
        self.dsp = dsp
        self.scene = scene
        self.fs = int(fe.fs_hz)
        self.fc_hz = fe.fc_hz
        self.hop = self.fs // 100
        self.m = 2 * self.hop
        base = base or Path.cwd()
        prof = base / sc.profile if sc.profile else None
        self.noise_spec = noise_spec(prof, load_band(sc.band), fe.fc_hz, fe.fs_hz, fe.noise_sigma_counts, sc.profile)
        self.front = Frontend(self.noise_spec, scene.t_sys_k, self.fs, self.hop, self.m, scene.rng.scope(), self.dsp)
        self.noise_on = noise
        self.prefetch = prefetch
        self.emitters: list[EmitterRender] = []
        self.wide: list[WideRender] = []
        for src in scene.sources:
            params = {e.id: e.signal.params for e in src.config.emitters}
            for em in src.emitters:
                sig = make_signal(em.signal_kind, params[em.id], Keys(scene.rng, src.id, em.id), self.dsp)
                if sig.mode == "wideband":
                    self.wide.append(WideRender(em, sig, fe.fc_hz, self.fs, self.hop, CHUNK_HOPS))
                    continue
                for ch in range(len(em.channels_mhz)):  # one per frequency (LoRa channels)
                    self.emitters.append(EmitterRender(em, sig, fe.fc_hz, self.fs, self.hop, self.m, self.dsp, ch))
        self.times: dict[str, float] = dict.fromkeys(STAGES, 0.0)  # seconds spent per stage (bench)
        self._tails: OrderedDict[int, np.ndarray | None] = OrderedDict()
        self._sig: tuple[int, np.ndarray | None] | None = None
        self._nz: tuple[int, np.ndarray | None] | None = None
        self._iq: tuple[int, np.ndarray] | None = None
        self._u8: tuple[int, np.ndarray] | None = None
        self._bufs: dict[str, np.ndarray] = {}

    def _buf(self, name: str, n: int, dtype: type) -> np.ndarray:
        """A buffer kept for the Renderer's life (one per role), allocated on first use."""
        b = self._bufs.get(name)
        if b is None:
            b = self._bufs[name] = np.empty(n, dtype=dtype)
        return b

    @property
    def gain(self) -> float:
        """ADC gain: counts per sqrt(mW)."""
        return self.front.gain

    # --- per chunk ---

    def _items(self, c: int) -> list[tuple[EmitterRender, Stream]]:
        tic = time.perf_counter()
        items = [(er, st) for er in self.emitters if (st := er.stream(c)) is not None]
        self.times["control"] += time.perf_counter() - tic
        return items

    def _synth(
        self, items: list[tuple[EmitterRender, Stream]], first: int, b0: int, nb: int, out: np.ndarray | None = None
    ) -> np.ndarray | None:
        tic = time.perf_counter()
        y = synth_blocks(self.dsp, items, first, b0, nb, self.m, self.front.h_rx, out)
        self.times["synth"] += time.perf_counter() - tic
        return y

    def _keep_tail(self, c: int, tail: np.ndarray | None) -> None:
        self._tails[c] = tail
        while len(self._tails) > 2:
            self._tails.popitem(last=False)

    def _tail(self, c: int) -> np.ndarray | None:
        """Signal in the second half of block 100c + 99 (it overlaps hop 100c + 100), or None if silent."""
        if c in self._tails:
            return self._tails[c]
        y = self._synth(self._items(c), CHUNK_HOPS - 1, CHUNK_HOPS * (c + 1) - 1, 1)
        tail = None if y is None else y[self.hop :].copy()
        self._keep_tail(c, tail)
        return tail

    def signal(self, c: int) -> np.ndarray | None:
        """Signals of chunk c before noise and ADC (complex64, sqrt(mW), fs samples), or None when silent."""
        if self._sig is not None and self._sig[0] == c:
            return self._sig[1]
        self._prefetch(c)
        prev = self._tail(c - 1)
        n = CHUNK_HOPS * self.hop
        buf = self._buf("sig", n + self.hop, np.complex64)
        self._sig = None  # the buffer is about to change
        y = self._synth(self._items(c), 0, CHUNK_HOPS * c, CHUNK_HOPS, buf)
        sig: np.ndarray | None = None
        tail: np.ndarray | None = None
        if y is not None:
            sig, tail = y[:n], y[n:].copy()
        if prev is not None:
            if sig is None:
                sig = buf[:n]
                sig[:] = 0
            sig[: self.hop] += prev
        if self.wide:
            tic = time.perf_counter()
            for wr in self.wide:
                w = wr.chunk(c)
                if w is None:
                    continue
                if sig is None:
                    sig = buf[:n]
                    sig[:] = 0
                sig += w
            self.times["wideband"] += time.perf_counter() - tic
        self._keep_tail(c, tail)
        self._sig = (c, sig)
        return sig

    def _noise(self, c: int) -> np.ndarray | None:
        """Shaped noise of chunk c (complex64, sqrt(mW)), or None when noise is off."""
        if not self.noise_on:
            return None
        if self._nz is not None and self._nz[0] == c:
            return self._nz[1]
        n = CHUNK_HOPS * self.hop
        tic = time.perf_counter()
        y = self.front.noise(CHUNK_HOPS * c, CHUNK_HOPS, self._buf("noise", n, np.complex64))
        self.times["noise"] += time.perf_counter() - tic
        self._nz = (c, y)
        return y

    def chunk_iq(self, c: int) -> np.ndarray:
        """Chunk c before the ADC: signals plus noise (complex64, sqrt(mW), fs samples)."""
        if self._iq is not None and self._iq[0] == c:
            return self._iq[1]
        sig, nz = self.signal(c), self._noise(c)
        y = self._buf("iq", CHUNK_HOPS * self.hop, np.complex64)
        if sig is not None and nz is not None:
            np.add(sig, nz, out=y)
        elif sig is not None or nz is not None:
            y[:] = sig if nz is None else nz
        else:
            y[:] = 0
        self._iq = (c, y)
        return y

    def chunk_u8(self, c: int) -> np.ndarray:
        """Chunk c as the uint8 stream (interleaved I, Q; 2 fs bytes)."""
        if self._u8 is not None and self._u8[0] == c:
            return self._u8[1]
        sig, nz = self.signal(c), self._noise(c)
        n = CHUNK_HOPS * self.hop
        out = self._buf("u8", 2 * n, np.uint8)
        tic = time.perf_counter()
        if sig is not None and nz is not None:
            u = self.front.adc(sig, nz, out)
        else:
            x = sig if sig is not None else nz
            u = self.front.adc(x if x is not None else np.zeros(n, dtype=np.complex64), out=out)
        self.times["adc"] += time.perf_counter() - tic
        self._u8 = (c, u)
        return u

    def _prefetch(self, c: int) -> None:
        """Live serving: in the last PREFETCH_S of an hour, compute the next hour's passes, a few emitters a chunk."""
        if not self.prefetch or not (self.emitters or self.wide):
            return
        t = c * CHUNK_HOPS * self.hop // self.fs  # Unix second of chunk c
        if t % BLOCK_S < BLOCK_S - PREFETCH_S:
            return
        nxt = t // BLOCK_S + 1
        tic = time.perf_counter()
        for sch in self._event_schedules():  # next hour's events (a VDL station draws ~9000), one per chunk
            if not sch.has_slot(nxt):
                sch.prefetch(nxt)
                break
        self.times["prefetch"] += time.perf_counter() - tic
        todo = [er for er in self.emitters if not er.fixed and not er.has_hour(nxt)]
        n_sat = sum(1 for er in self.emitters if not er.fixed)
        per = max(1, math.ceil(n_sat / (PREFETCH_S / 2)))  # done within the first half minute
        tic = time.perf_counter()
        for er in todo[:per]:
            er.prefetch_hour(nxt)
        self.times["prefetch"] += time.perf_counter() - tic

    def _event_schedules(self) -> list[Any]:
        out, seen = [], set()
        for em in [er.em for er in self.emitters] + [wr.em for wr in self.wide]:
            sch = em.schedule
            if sch.is_event and id(sch) not in seen:
                seen.add(id(sch))
                out.append(sch)
        return out

    # --- by absolute sample index ---

    def _chunks(self, n0: int, n: int) -> Iterator[tuple[int, int, int]]:
        """(chunk, first, last) sample offsets within each chunk covering absolute samples [n0, n0 + n)."""
        if n <= 0:
            return
        size = CHUNK_HOPS * self.hop
        for c in range(n0 // size, (n0 + n - 1) // size + 1):
            lo = max(n0, c * size) - c * size
            hi = min(n0 + n, (c + 1) * size) - c * size
            yield c, lo, hi

    def iq(self, n0: int, n: int) -> np.ndarray:
        """Absolute samples [n0, n0 + n) before the ADC (complex64, sqrt(mW))."""
        out = np.empty(max(n, 0), dtype=np.complex64)
        for c, lo, hi in self._chunks(n0, n):
            a = c * CHUNK_HOPS * self.hop + lo - n0
            out[a : a + hi - lo] = self.chunk_iq(c)[lo:hi]
        return out

    def counts(self, n0: int, n: int) -> np.ndarray:
        """As iq, scaled to ADC counts around 127.5 (not quantised or clipped)."""
        return self.iq(n0, n) * np.float32(self.gain)

    def iter_u8(self, n0: int, n: int) -> Iterator[np.ndarray]:
        """The uint8 stream (interleaved I, Q) for absolute samples [n0, n0 + n), one piece per chunk touched.

        Each piece is a view of the Renderer's buffer, valid until the next piece is requested (copy to keep it).
        """
        for c, lo, hi in self._chunks(n0, n):
            yield self.chunk_u8(c)[2 * lo : 2 * hi]

    def u8(self, n0: int, n: int) -> np.ndarray:
        """Absolute samples [n0, n0 + n) as the uint8 stream (a new array)."""
        out = np.empty(2 * max(n, 0), dtype=np.uint8)
        a = 0
        for piece in self.iter_u8(n0, n):
            out[a : a + len(piece)] = piece
            a += len(piece)
        return out
