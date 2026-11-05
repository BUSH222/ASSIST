// Philox4x32-10 (Salmon et al., SC 2011), the counter-based generator numpy's Philox also implements. Used for the
// wideband noise (frontend.cpp) and the random payload bytes of the modulated signals (modem.cpp).
// numpy twin: src/assist/dsp_ref/frontend.py (_philox).
#pragma once

#include <cstdint>

namespace assist {

inline void philox4x32_10(std::uint32_t c[4], std::uint32_t k0, std::uint32_t k1) {
    for (int r = 0; r < 10; ++r) {
        const std::uint64_t p0 = std::uint64_t(0xD2511F53u) * c[0];
        const std::uint64_t p1 = std::uint64_t(0xCD9E8D57u) * c[2];
        const std::uint32_t n0 = std::uint32_t(p1 >> 32) ^ c[1] ^ k0;
        const std::uint32_t n1 = std::uint32_t(p1);
        const std::uint32_t n2 = std::uint32_t(p0 >> 32) ^ c[3] ^ k1;
        const std::uint32_t n3 = std::uint32_t(p0);
        c[0] = n0;
        c[1] = n1;
        c[2] = n2;
        c[3] = n3;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
}

}  // namespace assist
