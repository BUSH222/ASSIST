// METEOR LRPT channel coding (plan §5). numpy twin: src/assist/dsp_ref/conv.py
#include "conv.hpp"

namespace assist {
namespace {

constexpr unsigned POLY_A = 0x4F;  // 171 octal, bit-reversed for a register with the newest bit in bit 0
constexpr unsigned POLY_B = 0x6D;  // 133 octal

inline std::uint8_t parity(unsigned v) {
    v ^= v >> 4;
    v ^= v >> 2;
    v ^= v >> 1;
    return static_cast<std::uint8_t>(v & 1u);
}

}  // namespace

void conv_encode(const std::uint8_t* bits, std::ptrdiff_t n, unsigned state, std::uint8_t* out) {
    unsigned sr = state & 0x3Fu;
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        sr = ((sr << 1) | (bits[i] & 1u)) & 0x7Fu;
        out[2 * i] = parity(sr & POLY_A);
        out[2 * i + 1] = parity(sr & POLY_B);
    }
}

void ccsds_randomise(const std::uint8_t* in, std::ptrdiff_t n, std::uint8_t* out) {
    // b[k + 8] = b[k + 7] ^ b[k + 5] ^ b[k + 3] ^ b[k], b[0..7] = 1. The register holds b[k .. k + 7] with b[k] in
    // bit 7, so each output bit is the register's top bit.
    unsigned sr = 0xFFu;
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        unsigned byte = 0;
        for (int j = 0; j < 8; ++j) {
            const unsigned b0 = (sr >> 7) & 1u;
            const unsigned fb = b0 ^ ((sr >> 4) & 1u) ^ ((sr >> 2) & 1u) ^ (sr & 1u);
            byte = (byte << 1) | b0;
            sr = ((sr << 1) | fb) & 0xFFu;
        }
        out[i] = static_cast<std::uint8_t>(in[i] ^ byte);
    }
}

}  // namespace assist
