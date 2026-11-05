"""Kernel backend selection.

`dsp` is the C++ module `assist._dsp`, or the numpy twins in `assist.dsp_ref` when the environment
variable ASSIST_PURE is set to anything other than "", "0", "false" or "no". Both expose the same
functions, so callers just use `from assist.backend import dsp`. There is no silent fallback: if the
C++ module is missing, importing this module fails unless ASSIST_PURE is set.
"""

import os

PURE: bool = os.environ.get("ASSIST_PURE", "").strip().lower() not in {"", "0", "false", "no"}

if PURE:
    from assist import dsp_ref as dsp
else:
    from assist import _dsp as dsp

NAME: str = "numpy" if PURE else "c++"

__all__ = ["NAME", "PURE", "dsp"]
