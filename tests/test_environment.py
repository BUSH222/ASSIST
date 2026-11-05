"""Interpreter and dependency checks from plan §2 (Python 3.15 checks)."""

import importlib
import sys
import sysconfig

import pytest


def test_python_315_regular_build() -> None:
    assert sys.version_info >= (3, 15)
    # Regular (GIL) build, not free-threaded 3.15t: the C++ kernels release the GIL themselves.
    assert not sysconfig.get_config_var("Py_GIL_DISABLED")


def test_sgp4_accelerated() -> None:
    # Without its C extension sgp4 silently falls back to pure Python and Skyfield gets far slower.
    from sgp4.api import accelerated

    assert accelerated is True


@pytest.mark.parametrize("module", ["numpy", "scipy", "skyfield", "sgp4", "pydantic", "yaml", "typer", "PIL"])
def test_runtime_dependency_imports(module: str) -> None:
    importlib.import_module(module)
