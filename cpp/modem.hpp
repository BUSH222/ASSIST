// Modulation kernels: random payload bytes, pulse shaping, LoRa chirps, the carrier NCO.
// numpy twin: src/assist/dsp_ref/modem.py
#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>

namespace assist {

using cf32 = std::complex<float>;

// n random bytes of block `block`: Philox4x32-10 with key `key` and counter (i, 0, block mod 2^32, block div 2^32)
// gives bytes 16i .. 16i + 15 (its four words, little endian). Any block is drawn on its own.
void random_bytes(std::uint64_t key, std::uint64_t block, std::ptrdiff_t n, std::uint8_t* out);

// Pulse shaping at sps samples per symbol. Symbol k (absolute index) is si[k - k0] + j sq[k - k0] (symbols outside
// the arrays are 0); h has ntaps (odd) taps centred on c = (ntaps - 1) / 2. Output sample n = n0 + j:
//   I(n) = sum_k si[k - k0] h[n - k sps + c],   Q(n) = sum_k sq[k - k0] h[n - qd - k sps + c]
// (qd = sps / 2 gives OQPSK's half-symbol Q delay, 0 a plain complex symbol stream). Sums in double.
void shape(const float* si, const float* sq, std::ptrdiff_t nsym, std::int64_t k0, const float* h,
           std::ptrdiff_t ntaps, std::int64_t sps, std::int64_t qd, std::int64_t n0, std::ptrdiff_t n, cf32* out);

// A LoRa burst (unit amplitude) as a sequence of chirp segments, starts increasing (a segment ends where the next
// begins, up to rounding): segment s starts at start[s] seconds after the burst start and lasts len[s] (a full
// symbol T = 2^sf / bw, or a fraction of one). val[s] >= 0: an up-chirp starting val * bw / 2^sf above -bw/2 and
// wrapping at +bw/2; val[s] < 0: a down-chirp from +bw/2. Each segment's phase starts at 0 cycles; full symbols
// (and the quarter down-chirp) end on whole cycles, so the phase is continuous. Sample j is at t0 + j dt seconds
// after the burst start; outside every segment it is 0.
void chirp(const double* start, const double* len, const std::int32_t* val, std::ptrdiff_t nseg, int sf, double bw,
           double t0, double dt, std::ptrdiff_t n, cf32* out);

// Carrier NCO and level: out[j] = x[j] * gain[j] * exp(2 pi i cyc[j]) (x may be null: a constant 1). The phase is
// reduced to [0, 1) cycles first; computed in double.
void nco(const cf32* x, const double* gain, const double* cyc, std::ptrdiff_t n, cf32* out);

}  // namespace assist
