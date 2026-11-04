// Complex float FFTs with pocketfft (cpp/third_party). numpy twin: scipy.fft.
#pragma once

#include <complex>
#include <cstddef>

#include "pocketfft_hdronly.h"

namespace assist {

using cf32 = std::complex<float>;

// `count` contiguous transforms of length n, in place. Forward: scaled by fct (1 = none); backward with fct = 1/n
// is the inverse DFT. pocketfft runs several transforms at once in SIMD lanes (~2.2x faster than one at a time for
// the sizes used here), and each transform's result is the same bits whichever lane or batch it is in, or alone.
// Plans come from pocketfft's own cache (POCKETFFT_CACHE_SIZE, set in CMakeLists.txt).
inline void fft_batch(cf32* data, std::size_t n, std::size_t count, bool forward, float fct) {
    if (count == 0) {
        return;
    }
    const pocketfft::shape_t shape{count, n};
    const pocketfft::stride_t stride{static_cast<std::ptrdiff_t>(n * sizeof(cf32)),
                                     static_cast<std::ptrdiff_t>(sizeof(cf32))};
    pocketfft::c2c(shape, stride, stride, {1}, forward, data, data, fct, 1);
}

}  // namespace assist
