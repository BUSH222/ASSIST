// pybind11 module assist._dsp. Type stubs: src/assist/_dsp.pyi
// Every binding releases the GIL around the kernel call (plan §2).
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>

#include "basic.hpp"

namespace py = pybind11;

using assist::cf32;
using CArray = py::array_t<cf32, py::array::c_style | py::array::forcecast>;

namespace {

CArray scale(const CArray& x, std::complex<double> g) {
    if (x.ndim() != 1) {
        throw std::invalid_argument("x must be 1-D");
    }
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

}  // namespace

PYBIND11_MODULE(_dsp, m) {
    m.doc() = "ASSIST C++ DSP kernels (numpy twins in assist.dsp_ref)";
    m.def("scale", &scale, py::arg("x"), py::arg("g"), "x * g as complex64 (x: 1-D, cast to complex64).");
}
