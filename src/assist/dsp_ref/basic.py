"""Twins of cpp/basic.hpp."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def scale(x: ArrayLike, g: complex) -> NDArray[np.complex64]:
    """x * g as complex64 (x: 1-D, cast to complex64)."""
    xa = np.ascontiguousarray(x, dtype=np.complex64)
    if xa.ndim != 1:
        raise ValueError("x must be 1-D")
    return xa * np.complex64(g)
