"""Wideband synthesis by overlap-add in the frequency domain; the kernel is `dsp.synth`.

Hop N = fs / 100 (10 ms), FFT size M = 2N, bin spacing fs / M = 50 Hz. Block b covers the absolute wideband samples
[bN, bN + 2N). An emitter at narrow rate r has L = r M / fs samples per block (even, since r is a multiple of 100):
narrow samples [bL/2, bL/2 + L) cover exactly the same time span.

Per block and emitter: v = x * hann_periodic(L) (x: the emitter's narrow samples, control already applied);
V = FFT_L(v); V * (M / L) * (-1)^(k b) is added to the accumulator at bins k - L/2 ... k + L/2 - 1 (mod M), where k is
the emitter's coarse bin (f_bin = fc + k * 50 Hz). The (-1)^(k b) factor keeps the coarse shift phase-continuous from
hop to hop, referred to the Unix epoch. Then A *= H_rx and y = IFFT_M(A); the periodic Hann at hop L/2 sums to one,
so hop h of the output is y_{h-1}[N:] + y_h[:N].

The engine synthesises one 1 s chunk (100 blocks) per call. Block 100c - 1, which hop 100c also needs, belongs to
chunk c - 1 (its control and narrow samples); its second half is carried over from the previous chunk, or computed on
its own when the render starts at chunk c. Both give the same bits: blocks are independent, and every transform
gives the same result whichever batch it runs in.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

    from assist.engine.control import EmitterRender, Stream


def hann(n: int) -> np.ndarray:
    """Periodic Hann window (float64): w[j] + w[j + n/2] = 1. The kernels use it rounded to float32."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def synth_blocks(
    dsp: ModuleType,
    items: list[tuple[EmitterRender, Stream]],
    first: int,
    b0: int,
    nb: int,
    m: int,
    h_rx: np.ndarray | None,
    out: np.ndarray | None = None,
) -> np.ndarray | None:
    """Blocks b0 .. b0 + nb - 1 from streams whose block `first` is b0, overlap-added: (nb + 1) * m / 2 samples
    (written into `out` if given).

    None when no emitter is active in any of the blocks (`out` is then left as it was).
    """
    streams, lens, bins, active = [], [], [], []
    for er, st in items:
        a = st.active[first : first + nb]
        if not a.any():
            continue
        half = er.n // 2
        streams.append(st.x[first * half : (first + nb + 1) * half])
        lens.append(er.n)
        bins.append(er.k_bin)
        active.append(a)
    if not streams:
        return None
    return dsp.synth(streams, lens, bins, active, b0, nb, m, h_rx, out)
