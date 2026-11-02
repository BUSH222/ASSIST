// Small element-wise kernels. numpy twin: src/assist/dsp_ref/basic.py
#pragma once

#include <complex>
#include <cstddef>

namespace assist {

using cf32 = std::complex<float>;

// out[i] = x[i] * g. The product is written out by hand so the compiler never
// calls the C99 Annex G NaN/inf recovery routine that std::complex uses.
inline void scale(const cf32* x, cf32* out, std::ptrdiff_t n, cf32 g) {
    const float gr = g.real();
    const float gi = g.imag();
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const float xr = x[i].real();
        const float xi = x[i].imag();
        out[i] = cf32(xr * gr - xi * gi, xr * gi + xi * gr);
    }
}

}  // namespace assist
