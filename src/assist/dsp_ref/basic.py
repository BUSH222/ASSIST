"""Twins of cpp/basic.hpp."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def output(out: np.ndarray | None, n: int, dtype: type) -> np.ndarray:
    """The output array of a kernel: `out` if given (checked as the C++ bindings check it), else a new one."""
    if out is None:
        return np.empty(n, dtype=dtype)
    if not isinstance(out, np.ndarray) or out.dtype != np.dtype(dtype):
        raise ValueError("out has the wrong dtype")
    if out.ndim != 1 or out.shape[0] != n:
        raise ValueError(f"out must be 1-D with {n} elements")
    if not (out.flags.c_contiguous and out.flags.writeable):
        raise ValueError("out must be C-contiguous and writable")
    return out


def scale(x: ArrayLike, g: complex) -> NDArray[np.complex64]:
    """x * g as complex64 (x: 1-D, cast to complex64)."""
    xa = np.ascontiguousarray(x, dtype=np.complex64)
    if xa.ndim != 1:
        raise ValueError("x must be 1-D")
    return xa * np.complex64(g)
