// Receiver front end kernels (plan §4.3, §4.4): white noise, noise shaping, ADC.
// numpy twin: src/assist/dsp_ref/frontend.py
#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>

namespace assist {

using cf32 = std::complex<float>;

// Complex white Gaussian noise, unit variance per component: samples [i0, i1) of hop h (i0, i1 even).
// Counter-based, so any hop (or part of one) is drawn on its own: samples 2j and 2j + 1 of hop h come from
// Philox4x32-10 with key `key` and counter (j, 0, h mod 2^32, h div 2^32). Each sample is one Box-Muller pair
// (I, Q) = r (x, y), r = sqrt(-2 ln u1), with ln, sin and cos evaluated by polynomials from + - * / and sqrt only,
// so the values are the same bits on every machine and in the numpy twin.
void white(std::uint64_t key, std::uint64_t h, std::ptrdiff_t i0, std::ptrdiff_t i1, cf32* out);

// Shaped noise of hops h0 .. h0 + nhops - 1 (h0 >= 1) into out[nhops * hop], by overlap-save in segments of `sub`
// samples (sub divides hop): segment [s, s + sub) of hop h is IFFT(FFT(row) * taps_f)[n_hist : n_hist + sub], where
// row (length ols = len(taps_f), zero-padded) holds the n_hist white samples before it (continuing into hop h - 1),
// then its sub samples. The result does not depend on which hops are computed together.
void noise(std::uint64_t key, std::uint64_t h0, std::ptrdiff_t nhops, std::ptrdiff_t hop, std::ptrdiff_t sub,
           const cf32* taps_f, std::ptrdiff_t ols, std::ptrdiff_t n_hist, cf32* out);

// ADC: out[2i], out[2i + 1] = clip(rint(127.5 + gain * (x + add)_I/Q), 0, 255); rint rounds half to even.
// `add` may be null.
void adc(const cf32* x, const cf32* add, std::ptrdiff_t n, float gain, std::uint8_t* out);

}  // namespace assist
