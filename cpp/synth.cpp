// Wideband synthesis. numpy twin: src/assist/dsp_ref/synth.py
#include "synth.hpp"

#include <algorithm>
#include <cmath>
#include <map>

#include "fft.hpp"

namespace assist {
namespace {

// M-point inverse transforms one at a time: a batch of 4 (SIMD lanes) is faster alone, but needs 1.5 MB, and six
// worker processes then overflow the shared L3 cache (measured: 1.9x real time each instead of 3.8x).
constexpr std::ptrdiff_t IFFT_BATCH = 1;

// Periodic Hann window w[j] = 0.5 - 0.5 cos(2 pi j / n), computed in double and rounded to float.
std::vector<float> hann(std::ptrdiff_t n) {
    std::vector<float> w(static_cast<std::size_t>(n));
    const double pi = 3.14159265358979323846;
    for (std::ptrdiff_t j = 0; j < n; ++j) {
        w[static_cast<std::size_t>(j)] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * pi * double(j) / double(n)));
    }
    return w;
}

// acc[k] += src[j] * scale * h_rx[k] for k = (first + j) mod m, j = 0 .. count - 1 (count <= m; h_rx may be null).
void add_scaled(cf32* acc, std::ptrdiff_t m, const cf32* src, std::ptrdiff_t count, std::int64_t first, float scale,
                const float* h_rx) {
    std::ptrdiff_t start = static_cast<std::ptrdiff_t>(((first % m) + m) % m);
    std::ptrdiff_t done = 0;
    while (done < count) {
        const std::ptrdiff_t run = std::min(count - done, m - start);
        float* a = reinterpret_cast<float*>(acc + start);
        const float* s = reinterpret_cast<const float*>(src + done);
        if (h_rx == nullptr) {
            for (std::ptrdiff_t j = 0; j < 2 * run; ++j) {
                a[j] += s[j] * scale;
            }
        } else {
            const float* h = h_rx + start;
            for (std::ptrdiff_t j = 0; j < run; ++j) {
                a[2 * j] += s[2 * j] * scale * h[j];
                a[2 * j + 1] += s[2 * j + 1] * scale * h[j];
            }
        }
        done += run;
        start = 0;
    }
}

template <typename T>
void grow(std::vector<T>& v, std::size_t n) {
    if (v.size() < n) {
        v.resize(n);
    }
}

// One emitter's windowed block spectra: FFT_L(x_block * hann_L) for its active blocks, slot[i] = row of block i.
// Kept per thread and reused (grow only): fresh pages for every chunk cost page faults.
struct Spectra {
    std::vector<cf32> rows;
    std::vector<std::ptrdiff_t> slot;  // -1: block not active
};

void block_spectra(const NarrowStream& e, std::ptrdiff_t nb, const std::vector<float>& w, Spectra& s) {
    const std::ptrdiff_t len = e.len;
    const std::ptrdiff_t half = len / 2;
    s.slot.assign(static_cast<std::size_t>(nb), -1);
    std::ptrdiff_t count = 0;
    for (std::ptrdiff_t i = 0; i < nb; ++i) {
        if (e.active[i]) {
            s.slot[static_cast<std::size_t>(i)] = count++;
        }
    }
    grow(s.rows, static_cast<std::size_t>(count * len));
    for (std::ptrdiff_t i = 0; i < nb; ++i) {
        const std::ptrdiff_t r = s.slot[static_cast<std::size_t>(i)];
        if (r < 0) {
            continue;
        }
        const float* src = reinterpret_cast<const float*>(e.x + i * half);
        float* dst = reinterpret_cast<float*>(s.rows.data() + r * len);
        for (std::ptrdiff_t j = 0; j < len; ++j) {
            dst[2 * j] = src[2 * j] * w[static_cast<std::size_t>(j)];
            dst[2 * j + 1] = src[2 * j + 1] * w[static_cast<std::size_t>(j)];
        }
    }
    fft_batch(s.rows.data(), static_cast<std::size_t>(len), static_cast<std::size_t>(count), true, 1.0f);
}

}  // namespace

std::ptrdiff_t synth(const std::vector<NarrowStream>& emitters, std::int64_t b0, std::ptrdiff_t nb, std::ptrdiff_t m,
                     const float* h_rx, cf32* out) {
    const std::ptrdiff_t hop = m / 2;
    std::fill(out, out + (nb + 1) * hop, cf32(0.0f, 0.0f));
    thread_local std::map<std::ptrdiff_t, std::vector<float>> windows;
    for (const auto& e : emitters) {
        if (!windows.count(e.len)) {
            windows[e.len] = hann(e.len);
        }
    }
    thread_local std::vector<Spectra> spectra;
    if (spectra.size() < emitters.size()) {
        spectra.resize(emitters.size());
    }
    for (std::size_t e = 0; e < emitters.size(); ++e) {
        block_spectra(emitters[e], nb, windows[emitters[e].len], spectra[e]);
    }
    // Blocks with at least one active emitter, synthesised IFFT_BATCH at a time.
    std::vector<std::ptrdiff_t> blocks;
    for (std::ptrdiff_t i = 0; i < nb; ++i) {
        for (std::size_t e = 0; e < emitters.size(); ++e) {
            if (spectra[e].slot[static_cast<std::size_t>(i)] >= 0) {
                blocks.push_back(i);
                break;
            }
        }
    }
    thread_local std::vector<cf32> acc;
    grow(acc, static_cast<std::size_t>(IFFT_BATCH * m));
    for (std::size_t g = 0; g < blocks.size(); g += IFFT_BATCH) {
        const std::size_t count = std::min<std::size_t>(IFFT_BATCH, blocks.size() - g);
        std::fill(acc.begin(), acc.begin() + static_cast<std::ptrdiff_t>(count) * m, cf32(0.0f, 0.0f));
        for (std::size_t r = 0; r < count; ++r) {
            const std::ptrdiff_t i = blocks[g + r];
            const std::int64_t b = b0 + i;
            cf32* a = acc.data() + static_cast<std::ptrdiff_t>(r) * m;
            for (std::size_t e = 0; e < emitters.size(); ++e) {
                const std::ptrdiff_t row = spectra[e].slot[static_cast<std::size_t>(i)];
                if (row < 0) {
                    continue;
                }
                const std::ptrdiff_t len = emitters[e].len;
                const std::ptrdiff_t half = len / 2;
                const std::int64_t k = emitters[e].k_bin;
                const cf32* spec = spectra[e].rows.data() + row * len;
                // (-1)^(k b): the coarse shift stays phase-continuous across hops (referred to the Unix epoch).
                const bool odd = ((k & 1) != 0) && ((b & 1) != 0);
                const float scale = static_cast<float>(double(m) / double(len)) * (odd ? -1.0f : 1.0f);
                // spec[half:] are bins -half .. -1, spec[:half] are bins 0 .. half - 1 (relative to k).
                add_scaled(a, m, spec + half, half, k - half, scale, h_rx);
                add_scaled(a, m, spec, half, k, scale, h_rx);
            }
        }
        fft_batch(acc.data(), static_cast<std::size_t>(m), count, false, 1.0f / static_cast<float>(m));
        for (std::size_t r = 0; r < count; ++r) {
            const float* a = reinterpret_cast<const float*>(acc.data() + static_cast<std::ptrdiff_t>(r) * m);
            float* o = reinterpret_cast<float*>(out + blocks[g + r] * hop);
            for (std::ptrdiff_t j = 0; j < 2 * m; ++j) {
                o[j] += a[j];
            }
        }
    }
    return static_cast<std::ptrdiff_t>(blocks.size());
}

}  // namespace assist
