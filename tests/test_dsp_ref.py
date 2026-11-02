"""T-REF: every C++ kernel in assist._dsp matches its numpy twin in assist.dsp_ref (rtol 1e-5)."""

import os
import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest

from assist import _dsp, dsp_ref

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

RTOL = 1e-5


def _random_iq(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


def test_twins_expose_the_same_kernels() -> None:
    native = {name for name in dir(_dsp) if not name.startswith("_")}
    assert native == set(dsp_ref.__all__)


@pytest.mark.parametrize("g", [1.0, 0.3 - 1.7j, -2.5e-3 + 4e2j])
def test_scale_matches_reference(g: complex) -> None:
    x = _random_iq(100_003, seed=1)
    got = _dsp.scale(x, g)
    want = dsp_ref.scale(x, g)
    assert got.dtype == want.dtype == np.complex64
    assert got.shape == want.shape == x.shape
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=0)


@pytest.mark.parametrize(
    "x",
    [
        _random_iq(1001, seed=2)[::3],  # non-contiguous
        _random_iq(64, seed=3).astype(np.complex128),  # cast down to complex64
        [1.0, 2j, -3.0 + 0.5j],  # plain list
        np.zeros(0, dtype=np.complex64),  # empty
    ],
    ids=["strided", "complex128", "list", "empty"],
)
def test_scale_input_conversion(x: ArrayLike) -> None:
    got = _dsp.scale(x, 2 - 1j)
    want = dsp_ref.scale(x, 2 - 1j)
    assert got.dtype == np.complex64
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=0)


@pytest.mark.parametrize("impl", [_dsp.scale, dsp_ref.scale], ids=["c++", "numpy"])
def test_scale_rejects_2d(impl) -> None:
    with pytest.raises(ValueError, match="1-D"):
        impl(np.zeros((2, 2), dtype=np.complex64), 1.0)


@pytest.mark.parametrize(("value", "expected"), [("", "c++"), ("0", "c++"), ("1", "numpy"), ("yes", "numpy")])
def test_assist_pure_selects_backend(value: str, expected: str) -> None:
    env = {**os.environ, "ASSIST_PURE": value}
    out = subprocess.run(
        [sys.executable, "-c", "from assist import backend; print(backend.NAME, backend.dsp.__name__)"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert out[0] == expected
    assert out[1] == ("assist.dsp_ref" if expected == "numpy" else "assist._dsp")
