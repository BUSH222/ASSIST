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
#include "frontend.hpp"
#include "synth.hpp"

namespace py = pybind11;

using assist::cf32;
using CArray = py::array_t<cf32, py::array::c_style | py::array::forcecast>;
using FArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using U8Array = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;
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
}
