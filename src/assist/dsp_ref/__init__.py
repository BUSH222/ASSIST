"""numpy reference twins of the C++ kernels in assist._dsp.

One module per C++ source file (cpp/basic.hpp -> basic.py, ...). Each function has the same name,
signature and dtypes as its C++ counterpart; tests compare the two (T-REF).
"""

from assist.dsp_ref.basic import scale

__all__ = ["scale"]
