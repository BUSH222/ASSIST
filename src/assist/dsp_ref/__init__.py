"""numpy reference twins of the C++ kernels in assist._dsp.

One module per C++ source file (cpp/basic.hpp -> basic.py, cpp/synth.cpp -> synth.py, cpp/frontend.cpp ->
frontend.py, cpp/modem.cpp -> modem.py, cpp/conv.cpp -> conv.py). Each function has the same name, signature and
dtypes as its C++ counterpart; tests compare the two. `white`, `adc`, `random_bytes`, `conv_encode` and
`ccsds_randomise` match bit for bit; the float kernels agree to rounding (FFT-based ones to ~1e-7 of their peak).
"""

from assist.dsp_ref.basic import scale
from assist.dsp_ref.conv import ccsds_randomise, conv_encode
from assist.dsp_ref.frontend import adc, noise, white
from assist.dsp_ref.modem import chirp, nco, random_bytes, shape
from assist.dsp_ref.synth import synth

_native = False  # mirrors assist._dsp._native (no -march=native in numpy)

__all__ = [
    "adc",
    "ccsds_randomise",
    "chirp",
    "conv_encode",
    "nco",
    "noise",
    "random_bytes",
    "scale",
    "shape",
    "synth",
    "white",
]
