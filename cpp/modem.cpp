// Modulation kernels (plan §5). numpy twin: src/assist/dsp_ref/modem.py
#include "modem.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "philox.hpp"

namespace assist {
namespace {

constexpr double TWO_PI = 6.283185307179586476925286766559;

// floor(a / b) for b > 0.
inline std::int64_t floor_div(std::int64_t a, std::int64_t b) {
    const std::int64_t q = a / b;
    return (a % b != 0 && a < 0) ? q - 1 : q;
}

// One branch of the polyphase filter: out[j] = sum_i s[kb - i] h[r + i sps], with kb = floor((m + c) / sps) and
// r = m + c - kb sps for sample m = m0 + j. Symbols come from a zero-padded copy, so the inner loop has a fixed
// length (taps beyond ntaps are 0); the sum runs over i ascending, i.e. k from kb down, in double.
void branch(const float* s, std::ptrdiff_t nsym, std::int64_t k0, const float* h, std::ptrdiff_t ntaps,
            std::int64_t sps, std::int64_t m0, std::ptrdiff_t n, bool imag, cf32* out) {
    const std::int64_t c = (ntaps - 1) / 2;
    const std::ptrdiff_t taps_per_phase = static_cast<std::ptrdiff_t>((ntaps + sps - 1) / sps);
    thread_local std::vector<double> hp, sp;
    hp.assign(static_cast<std::size_t>(sps * taps_per_phase), 0.0);
    for (std::int64_t r = 0; r < sps; ++r) {
        for (std::ptrdiff_t i = 0; i < taps_per_phase; ++i) {
            const std::int64_t t = r + i * sps;
            hp[static_cast<std::size_t>(r * taps_per_phase + i)] = t < ntaps ? double(h[t]) : 0.0;
        }
    }
    // Padded symbols: sp[q] = s[q - pad] (0 outside), covering every kb - i the window touches.
    const std::int64_t kb_first = floor_div(m0 + c, sps);
    const std::int64_t kb_last = floor_div(m0 + n - 1 + c, sps);
    const std::int64_t q0 = kb_first - (taps_per_phase - 1);  // lowest symbol index used
    const std::size_t len = static_cast<std::size_t>(kb_last - q0 + 1);
    sp.assign(len, 0.0);
    for (std::int64_t k = std::max(q0, k0); k < std::min(kb_last + 1, k0 + nsym); ++k) {
        sp[static_cast<std::size_t>(k - q0)] = double(s[k - k0]);
    }
    // BLOCK outputs at a time, so their (independent) sums interleave instead of each waiting on add latency;
    // each output's own sum keeps the order i = 0, 1, ...
    constexpr std::ptrdiff_t BLOCK = 64;
    const double* hrs[BLOCK];
    const double* sks[BLOCK];
    double acc[BLOCK];
    std::int64_t kb = kb_first;
    std::int64_t r = m0 + c - kb * sps;
    for (std::ptrdiff_t j0 = 0; j0 < n; j0 += BLOCK) {
        const std::ptrdiff_t nb = std::min(BLOCK, n - j0);
        for (std::ptrdiff_t b = 0; b < nb; ++b) {
            hrs[b] = hp.data() + r * taps_per_phase;
            sks[b] = sp.data() + (kb - q0);
            acc[b] = 0.0;
            if (++r == sps) {
                r = 0;
                ++kb;
            }
        }
        for (std::ptrdiff_t i = 0; i < taps_per_phase; ++i) {
            for (std::ptrdiff_t b = 0; b < nb; ++b) {
                acc[b] += sks[b][-i] * hrs[b][i];
            }
        }
        for (std::ptrdiff_t b = 0; b < nb; ++b) {
            float* o = reinterpret_cast<float*>(out + j0 + b);
            o[imag ? 1 : 0] = static_cast<float>(acc[b]);
        }
    }
}

}  // namespace

void random_bytes(std::uint64_t key, std::uint64_t block, std::ptrdiff_t n, std::uint8_t* out) {
    const std::uint32_t k0 = std::uint32_t(key);
    const std::uint32_t k1 = std::uint32_t(key >> 32);
    for (std::ptrdiff_t i = 0; 16 * i < n; ++i) {
        std::uint32_t c[4] = {std::uint32_t(i), 0u, std::uint32_t(block), std::uint32_t(block >> 32)};
        philox4x32_10(c, k0, k1);
        std::uint8_t b[16];
        for (int w = 0; w < 4; ++w) {
            for (int j = 0; j < 4; ++j) {
                b[4 * w + j] = static_cast<std::uint8_t>(c[w] >> (8 * j));
            }
        }
        const std::ptrdiff_t m = std::min<std::ptrdiff_t>(16, n - 16 * i);
        std::memcpy(out + 16 * i, b, static_cast<std::size_t>(m));
    }
}

void shape(const float* si, const float* sq, std::ptrdiff_t nsym, std::int64_t k0, const float* h,
           std::ptrdiff_t ntaps, std::int64_t sps, std::int64_t qd, std::int64_t n0, std::ptrdiff_t n, cf32* out) {
    if (n <= 0) {
        return;
    }
    branch(si, nsym, k0, h, ntaps, sps, n0, n, false, out);
    branch(sq, nsym, k0, h, ntaps, sps, n0 - qd, n, true, out);
}

void chirp(const double* start, const double* len, const std::int32_t* val, std::ptrdiff_t nseg, int sf, double bw,
           double t0, double dt, std::ptrdiff_t n, cf32* out) {
    const double nn = std::ldexp(1.0, sf);  // 2^sf
    const double t_sym = nn / bw;
    const double slope = bw / t_sym;  // Hz per second
    std::ptrdiff_t s = 0;
    for (std::ptrdiff_t j = 0; j < n; ++j) {
        const double tau = t0 + double(j) * dt;
        while (s < nseg && tau >= start[s] + len[s]) {
            ++s;
        }
        if (s >= nseg || tau < start[s]) {  // after the last segment, or before the first
            out[j] = cf32(0.0f, 0.0f);
            continue;
        }
        const double u = tau - start[s];
        double cyc;
        if (val[s] >= 0) {
            const double v = double(val[s]);
            const double t_wrap = t_sym * (1.0 - v / nn);
            cyc = (v * bw / nn - 0.5 * bw) * u + 0.5 * slope * u * u;
            if (u > t_wrap) {
                cyc -= bw * (u - t_wrap);
            }
        } else {
            cyc = 0.5 * bw * u - 0.5 * slope * u * u;
        }
        const double ph = TWO_PI * (cyc - std::floor(cyc));
        out[j] = cf32(static_cast<float>(std::cos(ph)), static_cast<float>(std::sin(ph)));
    }
}

void nco(const cf32* x, const double* gain, const double* cyc, std::ptrdiff_t n, cf32* out) {
    for (std::ptrdiff_t j = 0; j < n; ++j) {
        const double ph = TWO_PI * (cyc[j] - std::floor(cyc[j]));
        double re = gain[j] * std::cos(ph);
        double im = gain[j] * std::sin(ph);
        if (x != nullptr) {
            const double xr = x[j].real(), xi = x[j].imag();
            const double r = xr * re - xi * im;
            im = xr * im + xi * re;
            re = r;
        }
        out[j] = cf32(static_cast<float>(re), static_cast<float>(im));
    }
}

}  // namespace assist
