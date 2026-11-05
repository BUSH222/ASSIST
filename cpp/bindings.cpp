// pybind11 module assist._dsp. Type stubs: src/assist/_dsp.pyi; numpy twins: src/assist/dsp_ref/.
// Every binding checks its arguments, then releases the GIL around the kernel call (plan §2).
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "basic.hpp"
#include "conv.hpp"
#include "frontend.hpp"
#include "modem.hpp"
#include "synth.hpp"

namespace py = pybind11;

using assist::cf32;
using CArray = py::array_t<cf32, py::array::c_style | py::array::forcecast>;
using FArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using U8Array = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;
using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using I32Array = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;
// Outputs written in place (`out=`) are taken as plain py::array and checked, never converted.

namespace {

void require(bool ok, const std::string& what) {
    if (!ok) {
        throw std::invalid_argument(what);
    }
}

void require_1d(const py::array& a, const char* name) { require(a.ndim() == 1, std::string(name) + " must be 1-D"); }

// The output array: `out` if given (checked: exact dtype, 1-D, C-contiguous, writable, n elements, so the kernel
// never writes into a converted copy), else a new one. Reusing buffers avoids fresh pages for every chunk.
template <typename T>
py::array output(const std::optional<py::array>& out, py::ssize_t n) {
    if (!out.has_value()) {
        return py::array_t<T>(n);
    }
    const py::array& a = *out;
    require(a.dtype().is(py::dtype::of<T>()), "out has the wrong dtype");
    require(a.ndim() == 1 && a.shape(0) == n, "out must be 1-D with " + std::to_string(n) + " elements");
    require((a.flags() & py::array::c_style) != 0 && a.writeable(), "out must be C-contiguous and writable");
    return a;
}

CArray scale(const CArray& x, std::complex<double> g) {
    require_1d(x, "x");
    const py::ssize_t n = x.shape(0);
    CArray out(n);
    const cf32* src = x.data();
    cf32* dst = out.mutable_data();
    {
        py::gil_scoped_release release;
        assist::scale(src, dst, n, cf32(g));
    }
    return out;
}

py::array synth(const std::vector<CArray>& streams, const std::vector<std::int64_t>& lens,
                const std::vector<std::int64_t>& bins, const std::vector<U8Array>& active, std::int64_t b0,
                std::int64_t nb, std::int64_t m, const std::optional<FArray>& h_rx,
                const std::optional<py::array>& out) {
    const std::size_t ne = streams.size();
    require(lens.size() == ne && bins.size() == ne && active.size() == ne,
            "streams, lens, bins and active must have the same length");
    require(nb >= 1, "nb must be >= 1");
    require(m >= 2 && m % 2 == 0, "m must be even and >= 2");
    std::vector<assist::NarrowStream> em;
    em.reserve(ne);
    for (std::size_t e = 0; e < ne; ++e) {
        require_1d(streams[e], "streams[i]");
        require_1d(active[e], "active[i]");
        const std::int64_t len = lens[e];
        require(len >= 2 && len % 2 == 0 && len <= m, "lens[i] must be even, >= 2 and <= m");
        require(streams[e].shape(0) >= (nb + 1) * (len / 2), "streams[i] must hold (nb + 1) * lens[i] / 2 samples");
        require(active[e].shape(0) >= nb, "active[i] must hold nb flags");
        em.push_back({streams[e].data(), static_cast<std::ptrdiff_t>(len), bins[e], active[e].data()});
    }
    const float* h = nullptr;
    if (h_rx.has_value()) {
        require_1d(*h_rx, "h_rx");
        require(h_rx->shape(0) == m, "h_rx must hold m values");
        h = h_rx->data();
    }
    py::array res = output<cf32>(out, (nb + 1) * (m / 2));
    cf32* o = static_cast<cf32*>(res.mutable_data());
    {
        py::gil_scoped_release release;
        assist::synth(em, b0, static_cast<std::ptrdiff_t>(nb), static_cast<std::ptrdiff_t>(m), h, o);
    }
    return res;
}

CArray white(std::uint64_t key, std::uint64_t h0, std::int64_t nhops, std::int64_t hop) {
    require(nhops >= 0, "nhops must be >= 0");
    require(hop >= 2 && hop % 2 == 0, "hop must be even and >= 2");
    CArray out(nhops * hop);
    cf32* o = out.mutable_data();
    {
        py::gil_scoped_release release;
        for (std::int64_t i = 0; i < nhops; ++i) {
            assist::white(key, h0 + std::uint64_t(i), 0, hop, o + i * hop);
        }
    }
    return out;
}

py::array noise(std::uint64_t key, std::uint64_t h0, std::int64_t nhops, std::int64_t hop, std::int64_t sub,
                const CArray& taps_f, std::int64_t n_hist, const std::optional<py::array>& out) {
    require_1d(taps_f, "taps_f");
    require(h0 >= 1, "h0 must be >= 1 (hop h0 - 1 supplies the history)");
    require(nhops >= 0, "nhops must be >= 0");
    require(hop >= 2 && hop % 2 == 0, "hop must be even and >= 2");
    require(sub >= 2 && sub % 2 == 0 && hop % sub == 0, "sub must be even and divide hop");
    require(n_hist >= 0 && n_hist % 2 == 0 && n_hist <= hop, "n_hist must be even and in [0, hop]");
    const std::int64_t ols = taps_f.shape(0);
    require(ols >= sub + n_hist, "taps_f must hold at least sub + n_hist bins");
    py::array res = output<cf32>(out, nhops * hop);
    cf32* o = static_cast<cf32*>(res.mutable_data());
    const cf32* tf = taps_f.data();
    {
        py::gil_scoped_release release;
        assist::noise(key, h0, static_cast<std::ptrdiff_t>(nhops), static_cast<std::ptrdiff_t>(hop),
                      static_cast<std::ptrdiff_t>(sub), tf, static_cast<std::ptrdiff_t>(ols),
                      static_cast<std::ptrdiff_t>(n_hist), o);
    }
    return res;
}

py::array adc(const CArray& x, double gain, const std::optional<CArray>& add, const std::optional<py::array>& out) {
    require_1d(x, "x");
    const py::ssize_t n = x.shape(0);
    const cf32* a = nullptr;
    if (add.has_value()) {
        require_1d(*add, "add");
        require(add->shape(0) == n, "add must have the same length as x");
        a = add->data();
    }
    py::array res = output<std::uint8_t>(out, 2 * n);
    std::uint8_t* o = static_cast<std::uint8_t*>(res.mutable_data());
    const cf32* src = x.data();
    {
        py::gil_scoped_release release;
        assist::adc(src, a, n, static_cast<float>(gain), o);
    }
    return res;
}

U8Array random_bytes(std::uint64_t key, std::uint64_t block, std::int64_t n) {
    require(n >= 0, "n must be >= 0");
    U8Array out(n);
    std::uint8_t* o = out.mutable_data();
    {
        py::gil_scoped_release release;
        assist::random_bytes(key, block, static_cast<std::ptrdiff_t>(n), o);
    }
    return out;
}

U8Array conv_encode(const U8Array& bits, unsigned state) {
    require_1d(bits, "bits");
    require(state < 64u, "state must hold 6 bits (0 .. 63)");
    const py::ssize_t n = bits.shape(0);
    U8Array out(2 * n);
    const std::uint8_t* b = bits.data();
    std::uint8_t* o = out.mutable_data();
    {
        py::gil_scoped_release release;
        assist::conv_encode(b, n, state, o);
    }
    return out;
}

U8Array ccsds_randomise(const U8Array& data) {
    require_1d(data, "data");
    const py::ssize_t n = data.shape(0);
    U8Array out(n);
    const std::uint8_t* d = data.data();
    std::uint8_t* o = out.mutable_data();
    {
        py::gil_scoped_release release;
        assist::ccsds_randomise(d, n, o);
    }
    return out;
}

CArray shape(const FArray& si, const FArray& sq, std::int64_t k0, const FArray& h, std::int64_t sps, std::int64_t qd,
             std::int64_t n0, std::int64_t n) {
    require_1d(si, "si");
    require_1d(sq, "sq");
    require_1d(h, "h");
    require(si.shape(0) == sq.shape(0), "si and sq must have the same length");
    require(h.shape(0) >= 1 && h.shape(0) % 2 == 1, "h must hold an odd number of taps");
    require(sps >= 1, "sps must be >= 1");
    require(qd >= 0 && qd < sps, "qd must be in [0, sps)");
    require(n >= 0, "n must be >= 0");
    CArray out(n);
    const float* a = si.data();
    const float* b = sq.data();
    const float* t = h.data();
    cf32* o = out.mutable_data();
    {
        py::gil_scoped_release release;
        assist::shape(a, b, si.shape(0), k0, t, h.shape(0), sps, qd, n0, static_cast<std::ptrdiff_t>(n), o);
    }
    return out;
}

CArray chirp(const DArray& start, const DArray& length, const I32Array& val, int sf, double bw, double t0, double dt,
             std::int64_t n) {
    require_1d(start, "start");
    require_1d(length, "length");
    require_1d(val, "val");
    require(start.shape(0) == length.shape(0) && start.shape(0) == val.shape(0), "start, length and val must match");
    require(start.shape(0) >= 1, "at least one segment is needed");
    require(sf >= 5 && sf <= 12, "sf must be in 5 .. 12");
    require(bw > 0 && dt > 0, "bw and dt must be > 0");
    require(n >= 0, "n must be >= 0");
    const std::int64_t ns = start.shape(0);
    const double* st = start.data();
    const double* ln = length.data();
    const std::int32_t* v = val.data();
    for (std::int64_t s = 0; s < ns; ++s) {
        require(ln[s] > 0, "segment lengths must be > 0");
        require(s == 0 || st[s] > st[s - 1], "segment starts must increase");
        require(v[s] < (1 << sf), "chirp values must be < 2^sf");
    }
    CArray out(n);
    cf32* o = out.mutable_data();
    {
        py::gil_scoped_release release;
        assist::chirp(st, ln, v, static_cast<std::ptrdiff_t>(ns), sf, bw, t0, dt, static_cast<std::ptrdiff_t>(n), o);
    }
    return out;
}

py::array nco(const std::optional<CArray>& x, const DArray& gain, const DArray& cyc,
              const std::optional<py::array>& out) {
    require_1d(gain, "gain");
    require_1d(cyc, "cyc");
    const py::ssize_t n = gain.shape(0);
    require(cyc.shape(0) == n, "gain and cyc must have the same length");
    const cf32* xp = nullptr;
    if (x.has_value()) {
        require_1d(*x, "x");
        require(x->shape(0) == n, "x must have the same length as gain");
        xp = x->data();
    }
    py::array res = output<cf32>(out, n);
    cf32* o = static_cast<cf32*>(res.mutable_data());
    const double* g = gain.data();
    const double* c = cyc.data();
    {
        py::gil_scoped_release release;
        assist::nco(xp, g, c, n, o);
    }
    return res;
}

}  // namespace

PYBIND11_MODULE(_dsp, m) {
    m.doc() = "ASSIST C++ DSP kernels (numpy twins in assist.dsp_ref)";
#ifdef ASSIST_NATIVE_BUILD
    m.attr("_native") = true;
#else
    m.attr("_native") = false;
#endif
    m.def("scale", &scale, py::arg("x"), py::arg("g"), "x * g as complex64 (x: 1-D, cast to complex64).");
    m.def("synth", &synth, py::arg("streams"), py::arg("lens"), py::arg("bins"), py::arg("active"), py::arg("b0"),
          py::arg("nb"), py::arg("m"), py::arg("h_rx") = py::none(), py::arg("out") = py::none(),
          "Overlap-add synthesis of blocks b0 .. b0 + nb - 1 into (nb + 1) * m / 2 samples (plan §4.2).");
    m.def("white", &white, py::arg("key"), py::arg("h0"), py::arg("nhops"), py::arg("hop"),
          "Complex white Gaussian noise (unit variance per component) of hops h0 .. h0 + nhops - 1.");
    m.def("noise", &noise, py::arg("key"), py::arg("h0"), py::arg("nhops"), py::arg("hop"), py::arg("sub"),
          py::arg("taps_f"),
          py::arg("n_hist"), py::arg("out") = py::none(), "Shaped noise of hops h0 .. h0 + nhops - 1 by overlap-save (plan §4.3).");
    m.def("adc", &adc, py::arg("x"), py::arg("gain"), py::arg("add") = py::none(), py::arg("out") = py::none(),
          "Interleaved uint8 I, Q: clip(rint(127.5 + gain * (x + add)), 0, 255).");
    m.def("random_bytes", &random_bytes, py::arg("key"), py::arg("block"), py::arg("n"),
          "n random bytes of block `block` (Philox4x32-10, counter (i, 0, block)).");
    m.def("conv_encode", &conv_encode, py::arg("bits"), py::arg("state") = 0,
          "K = 7, r = 1/2 convolutional code (171, 133 octal) of 0/1 bits; state = previous 6 input bits.");
    m.def("ccsds_randomise", &ccsds_randomise, py::arg("data"), "XOR with the CCSDS randomiser sequence (FF 48 0E ...).");
    m.def("shape", &shape, py::arg("si"), py::arg("sq"), py::arg("k0"), py::arg("h"), py::arg("sps"), py::arg("qd"),
          py::arg("n0"), py::arg("n"), "Pulse shaping of symbols k0 .. into samples n0 .. n0 + n - 1.");
    m.def("chirp", &chirp, py::arg("start"), py::arg("length"), py::arg("val"), py::arg("sf"), py::arg("bw"),
          py::arg("t0"), py::arg("dt"), py::arg("n"), "A LoRa burst from chirp segments, sampled at t0 + j dt.");
    m.def("nco", &nco, py::arg("x"), py::arg("gain"), py::arg("cyc"), py::arg("out") = py::none(),
          "x * gain * exp(2 pi i cyc) as complex64 (x None: a constant 1).");
}
