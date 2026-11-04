// Receiver front end kernels (plan §4.3, §4.4). numpy twin: src/assist/dsp_ref/frontend.py
#include "frontend.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "fft.hpp"

namespace assist {
namespace {

constexpr std::ptrdiff_t WORDS_BLOCK = 2048;  // samples per pass (16 KiB of Philox words, stays in L1)
constexpr std::ptrdiff_t OLS_BATCH = 4;       // overlap-save hops transformed together (SIMD lanes)

// clip(rint(127.5 + gain * y), 0, 255). For v in [0, 255], (v + 2^23) - 2^23 is rint(v) (round half to even), and
// clip(rint(v), 0, 255) = rint(clip(v, 0, 255)) since rint is monotonic and the bounds are integers.
inline std::uint8_t quantise(float y, float gain) {
    const float big = 8388608.0f;
    float v = y * gain + 127.5f;
    v = std::min(std::max(v, 0.0f), 255.0f);
    v = (v + big) - big;
    return static_cast<std::uint8_t>(static_cast<int>(v));
}

// Philox4x32-10 (Salmon et al., SC 2011), the counter-based generator numpy's Philox also implements.
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

// Natural log of x in (0, 1]: x = m 2^e with m in [sqrt(1/2), sqrt(2)), ln m = 2 atanh(s), s = (m - 1) / (m + 1),
// |s| <= 0.172, by its series to s^9 (error < 1e-9); ln 2 split in a high and a low part.
inline float ln_poly(float x) {
    std::uint32_t b;
    std::memcpy(&b, &x, sizeof b);
    int e = int((b >> 23) & 0xFFu) - 127;
    const std::uint32_t mb = (b & 0x7FFFFFu) | 0x3F800000u;
    float m;
    std::memcpy(&m, &mb, sizeof m);
    const bool big = m > 1.41421356f;
    m = big ? m * 0.5f : m;
    e = big ? e + 1 : e;
    const float s = (m - 1.0f) / (m + 1.0f);
    const float s2 = s * s;
    const float p =
        2.0f * s *
        (1.0f + s2 * (0x1.555556p-2f + s2 * (0x1.99999ap-3f + s2 * (0x1.24924ap-3f + s2 * 0x1.c71c72p-4f))));
    const float fe = float(e);
    return fe * 0x1.62e4p-1f + (fe * 0x1.7f7d1cp-20f + p);
}

// Box-Muller on word pairs: u1 = (wa + 0.5) 2^-32 in (0, 1); the angle is uniform on the circle: the top 2 bits of
// wb pick the quadrant (an exact rotation), the next 24 give t in [-pi/4, pi/4), where sin and cos are Taylor
// series to t^9 and t^10 (error < 2e-9).
void box_muller(const std::uint32_t* w, std::ptrdiff_t n, cf32* out) {
    float* o = reinterpret_cast<float*>(out);
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const std::uint32_t wa = w[2 * i];
        const std::uint32_t wb = w[2 * i + 1];
        const float u1 = (float(wa) + 0.5f) * 0x1p-32f;
        const float r = std::sqrt(-2.0f * ln_poly(u1));
        const float t = (float((wb >> 6) & 0xFFFFFFu) * 0x1p-24f - 0.5f) * 1.57079633f;
        const float t2 = t * t;
        const float sn =
            t * (1.0f + t2 * (-0x1.555556p-3f + t2 * (0x1.111112p-7f + t2 * (-0x1.a01a02p-13f + t2 * 0x1.71de3ap-19f))));
        const float cs =
            1.0f +
            t2 * (-0.5f + t2 * (0x1.555556p-5f + t2 * (-0x1.6c16c2p-10f + t2 * (0x1.a01a02p-16f - t2 * 0x1.27e4fcp-22f))));
        const std::uint32_t q = wb >> 30;
        const float x = (q == 0) ? cs : (q == 1) ? -sn : (q == 2) ? -cs : sn;
        const float y = (q == 0) ? sn : (q == 1) ? cs : (q == 2) ? -sn : -cs;
        o[2 * i] = r * x;
        o[2 * i + 1] = r * y;
    }
}

template <typename T>
void grow(std::vector<T>& v, std::size_t n) {
    if (v.size() < n) {
        v.resize(n);
    }
}

}  // namespace

void white(std::uint64_t key, std::uint64_t h, std::ptrdiff_t i0, std::ptrdiff_t i1, cf32* out) {
    const std::uint32_t k0 = std::uint32_t(key);
    const std::uint32_t k1 = std::uint32_t(key >> 32);
    const std::uint32_t h_lo = std::uint32_t(h);
    const std::uint32_t h_hi = std::uint32_t(h >> 32);
    thread_local std::vector<std::uint32_t> w(2 * WORDS_BLOCK);
    for (std::ptrdiff_t s0 = i0; s0 < i1; s0 += WORDS_BLOCK) {
        const std::ptrdiff_t s1 = std::min(i1, s0 + WORDS_BLOCK);
        for (std::ptrdiff_t j = s0 / 2; j < s1 / 2; ++j) {
            std::uint32_t c[4] = {std::uint32_t(j), 0u, h_lo, h_hi};
            philox4x32_10(c, k0, k1);
            std::memcpy(w.data() + 4 * (j - s0 / 2), c, sizeof c);
        }
        box_muller(w.data(), s1 - s0, out + (s0 - i0));
    }
}

void noise(std::uint64_t key, std::uint64_t h0, std::ptrdiff_t nhops, std::ptrdiff_t hop, std::ptrdiff_t sub,
           const cf32* taps_f, std::ptrdiff_t ols, std::ptrdiff_t n_hist, cf32* out) {
    // Overlap-save in segments of `sub` samples (hop / sub per hop), OLS_BATCH segments transformed together (SIMD
    // lanes). Each row is drawn straight from the counter-based generator: the n_hist white samples before the
    // segment (from hop h - 1 for a hop's first segment), then its own sub samples, then zeros. Short segments keep
    // a batch inside the core's L2 cache; the scratch rows are kept per thread and only grow.
    const std::ptrdiff_t per_hop = hop / sub;
    const std::ptrdiff_t nseg = nhops * per_hop;
    thread_local std::vector<cf32> seg;
    grow(seg, static_cast<std::size_t>(OLS_BATCH * ols));
    const float* tf = reinterpret_cast<const float*>(taps_f);
    for (std::ptrdiff_t g = 0; g < nseg; g += OLS_BATCH) {
        const std::ptrdiff_t count = std::min(OLS_BATCH, nseg - g);
        for (std::ptrdiff_t r = 0; r < count; ++r) {
            const std::ptrdiff_t q = g + r;
            const std::uint64_t h = h0 + std::uint64_t(q / per_hop);
            const std::ptrdiff_t s0 = (q % per_hop) * sub;
            cf32* row = seg.data() + r * ols;
            if (s0 >= n_hist) {
                white(key, h, s0 - n_hist, s0, row);
            } else {
                white(key, h - 1, hop - (n_hist - s0), hop, row);
                white(key, h, 0, s0, row + (n_hist - s0));
            }
            white(key, h, s0, s0 + sub, row + n_hist);
            std::fill(row + n_hist + sub, row + ols, cf32(0.0f, 0.0f));
        }
        fft_batch(seg.data(), static_cast<std::size_t>(ols), static_cast<std::size_t>(count), true, 1.0f);
        for (std::ptrdiff_t r = 0; r < count; ++r) {
            float* sf = reinterpret_cast<float*>(seg.data() + r * ols);
            for (std::ptrdiff_t k = 0; k < ols; ++k) {
                const float ar = sf[2 * k], ai = sf[2 * k + 1];
                const float br = tf[2 * k], bi = tf[2 * k + 1];
                sf[2 * k] = ar * br - ai * bi;
                sf[2 * k + 1] = ar * bi + ai * br;
            }
        }
        fft_batch(seg.data(), static_cast<std::size_t>(ols), static_cast<std::size_t>(count), false,
                  1.0f / static_cast<float>(ols));
        for (std::ptrdiff_t r = 0; r < count; ++r) {
            const cf32* row = seg.data() + r * ols;
            std::copy(row + n_hist, row + n_hist + sub, out + (g + r) * sub);
        }
    }
}

void adc(const cf32* x, const cf32* add, std::ptrdiff_t n, float gain, std::uint8_t* out) {
    const float* xf = reinterpret_cast<const float*>(x);
    if (add == nullptr) {
        for (std::ptrdiff_t j = 0; j < 2 * n; ++j) {
            out[j] = quantise(xf[j], gain);
        }
    } else {
        const float* af = reinterpret_cast<const float*>(add);
        for (std::ptrdiff_t j = 0; j < 2 * n; ++j) {
            out[j] = quantise(xf[j] + af[j], gain);
        }
    }
}

}  // namespace assist
