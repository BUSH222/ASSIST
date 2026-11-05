"""P4/P5: waterfall .raw loader and row timing."""

from typing import TYPE_CHECKING

import numpy as np
import pytest

from assist import waterfall as wf
from assist.profile.rawio import open_raw

if TYPE_CHECKING:
    from pathlib import Path


def test_open_raw_and_row_timing(tmp_path: Path) -> None:
    p = tmp_path / "waterfall_1790507202.raw"
    data = np.zeros((3, wf.FFT_SIZE), dtype=np.uint8)
    data[1, :] = 85
    data.tofile(p)
    raw = open_raw(p)
    assert raw.n_rows == 3
    assert raw.t_file == 1790507202
    # P5: t_row(i) = t_file + 11 993 088/fs + (i + 0.5) * 2 392 064/fs
    assert raw.row_time(0) == pytest.approx(1790507202 + 11_993_088 / 2.4e6 + 0.5 * 2_392_064 / 2.4e6, abs=1e-9)
    assert raw.row_time(2) - raw.row_time(1) == pytest.approx(0.99669, abs=1e-5)
    assert raw.row_at(float(raw.row_time(1))) == 1
    np.testing.assert_allclose(raw.db(1, 2), 5.0, atol=1e-5)


def test_open_raw_rejects_bad_files(tmp_path: Path) -> None:
    bad_name = tmp_path / "wf.raw"
    bad_name.write_bytes(bytes(wf.FFT_SIZE))
    with pytest.raises(ValueError, match="waterfall_"):
        open_raw(bad_name)
    bad_size = tmp_path / "waterfall_1.raw"
    bad_size.write_bytes(bytes(wf.FFT_SIZE + 1))
    with pytest.raises(ValueError, match="multiple"):
        open_raw(bad_size)
