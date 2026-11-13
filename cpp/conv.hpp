// METEOR LRPT channel coding (`oqpsk` framing `meteor_cadu`): CCSDS randomiser and the K = 7, r = 1/2
// convolutional encoder. numpy twin: src/assist/dsp_ref/conv.py
#pragma once

#include <cstddef>
#include <cstdint>

namespace assist {

// K = 7, r = 1/2 convolutional code with generators 171 and 133 (octal), as libfec's 0x4F / 0x6D on a register
// shifted left with the new bit in bit 0. `state` holds the previous 6 input bits (bit 0 = the most recent), so a
// frame can be encoded on its own from the end of the previous one. bits: n values 0/1; out: 2n values 0/1
// (G1 then G2 per input bit, neither inverted).
void conv_encode(const std::uint8_t* bits, std::ptrdiff_t n, unsigned state, std::uint8_t* out);

// CCSDS randomiser: out[i] = in[i] ^ pn[i], pn from h(x) = x^8 + x^7 + x^5 + x^3 + 1 with all ones loaded, read
// MSB first (FF 48 0E C0 9A ...). The sequence restarts at every call (one call per frame).
void ccsds_randomise(const std::uint8_t* in, std::ptrdiff_t n, std::uint8_t* out);

}  // namespace assist
