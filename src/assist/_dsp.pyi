"""Type stubs for the C++ module assist._dsp (cpp/bindings.cpp). Keep in sync with assist.dsp_ref."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

def scale(x: ArrayLike, g: complex) -> NDArray[np.complex64]:
    """x * g as complex64 (x: 1-D, cast to complex64)."""
