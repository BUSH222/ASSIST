"""numpy reference twins of the C++ kernels in assist._dsp.

One module per C++ source file (cpp/basic.hpp -> basic.py, cpp/synth.cpp -> synth.py, cpp/frontend.cpp ->
frontend.py). Each function has the same name, signature and dtypes as its C++ counterpart; tests compare the two
(T-REF). `white` matches bit for bit; FFT-based kernels agree to ~1e-7 of their output's peak.
"""

from assist.dsp_ref.basic import scale
from assist.dsp_ref.frontend import adc, noise, white
from assist.dsp_ref.synth import synth

_native = False  # mirrors assist._dsp._native (no -march=native in numpy)

__all__ = ["adc", "noise", "scale", "synth", "white"]
