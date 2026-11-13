// Wideband synthesis by overlap-add in the frequency domain. numpy twin: src/assist/dsp_ref/synth.py
#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace assist {

using cf32 = std::complex<float>;

// One emitter's narrow samples for blocks b0 .. b0 + nb - 1. Block b0 + i is x[i * len / 2, i * len / 2 + len),
// so x holds (nb + 1) * len / 2 samples.
struct NarrowStream {
    const cf32* x;
    std::ptrdiff_t len;          // L: narrow samples per block (even, <= M)
    std::int64_t k_bin;          // coarse bin: f_bin = fc + k_bin * fs / M (any sign; wraps mod M)
    const std::uint8_t* active;  // nb flags; block b0 + i is synthesised only where active[i] != 0
};

// Blocks b0 .. b0 + nb - 1 (each M samples, hop M / 2), overlap-added into out[(nb + 1) * M / 2]:
// block b0 + i lands at out[i * M / 2, i * M / 2 + M). Per block and active emitter (in order):
// V = FFT_L(x_block * hann_L) * (M / L) * (-1)^(k_bin * b) * h_rx (if not null), added at bins
// k_bin - L/2 .. k_bin + L/2 - 1 (mod M); then y = IFFT_M(A). Silent blocks are skipped. Returns the number of blocks
// synthesised.
std::ptrdiff_t synth(const std::vector<NarrowStream>& emitters, std::int64_t b0, std::ptrdiff_t nb, std::ptrdiff_t m,
                     const float* h_rx, cf32* out);

}  // namespace assist
