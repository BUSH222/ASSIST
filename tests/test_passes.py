"""Profile pass helper: OMM reading, time conversion, geometry and passes.

Uses tests/fixtures/omm_2026-11-02.csv (public elements from the 2026-11-02 retlector snapshots) and the plan's
rounded station position, so no local data is needed.
"""

import csv
import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from assist.profile import passes
from assist.profile.station import Station, load_station

FIXTURE = Path(__file__).parent / "fixtures" / "omm_2026-11-02.csv"
STATION = Station(55.5, 37.1, 180.0)
REC_START = dt.datetime(2026, 11, 1, 16, 39, tzinfo=dt.UTC).timestamp()  # the station recording
FM117, METEOR_M2_3 = 41188, 57166


@pytest.fixture(scope="module")
def els() -> dict[int, dict[str, str]]:
    return passes.load_elements([FIXTURE])


def test_unix_to_time_is_exact_across_leap_seconds() -> None:
    u = np.array([REC_START, REC_START + 0.25, 1483228799.0, 1483228800.0, 946684800.5])
    t = passes.unix_to_time(u)
    ref = passes.timescale().from_datetimes([dt.datetime.fromtimestamp(x, dt.UTC) for x in u])
    np.testing.assert_allclose((t.tt - ref.tt) * 86400, 0, atol=1e-5)


def test_newer_epoch_wins(els: dict[int, dict[str, str]]) -> None:
    assert els[FM117]["EPOCH"] == "2026-11-02T10:50:20.095872"
    assert els[METEOR_M2_3]["EPOCH"] == "2026-11-02T12:21:12.412800"
    assert len(els) == 3


def test_columns_by_name(tmp_path: Path) -> None:
    rows = list(csv.reader(FIXTURE.open()))
    order = list(reversed(range(len(rows[0]))))
    shuffled = tmp_path / "shuffled.csv"
    with shuffled.open("w", newline="") as f:
        csv.writer(f).writerows([[r[i] for i in order] for r in rows])
    assert passes.load_elements([shuffled]) == passes.load_elements([FIXTURE])
    missing = tmp_path / "missing.csv"
    with missing.open("w", newline="") as f:
        csv.writer(f).writerows([r[1:] for r in rows])
    with pytest.raises(ValueError, match="OBJECT_NAME"):
        passes.read_omm_csv(missing)


def test_range_rate_is_derivative_of_range(els: dict[int, dict[str, str]]) -> None:
    t = REC_START + 1500 + np.array([-0.5, 0.0, 0.5])
    g = passes.geometry(els[FM117], STATION, t)
    assert (g.range_m[2] - g.range_m[0]) / 1.0 == pytest.approx(g.range_rate_m_s[1], abs=0.01)
    assert g.range_rate_m_s[1] < 0  # rising: approaching
    assert g.doppler_hz(137.46e6)[1] > 0


def test_fm117_pass_in_the_recording(els: dict[int, dict[str, str]]) -> None:
    """Regression of the timestamp -> geometry chain: FM117 rises 1411 s into the recording (seen at ~1418 s)."""
    (p,) = passes.find_passes(els[FM117], STATION, REC_START, REC_START + 1574.24)
    assert p.aos - REC_START == pytest.approx(1411.0, abs=0.5)
    assert p.los == REC_START + 1574.24  # clipped: still rising at the end of the file
    assert p.max_el_deg == pytest.approx(10.7, abs=0.1)


def test_passes_over_a_day(els: dict[int, dict[str, str]]) -> None:
    ps = passes.find_passes(els[METEOR_M2_3], STATION, REC_START, REC_START + 86400)
    assert 6 <= len(ps) <= 12  # polar orbit seen from 55.5 N: about 10 passes a day
    for p in ps:
        assert p.aos < p.tca < p.los
        assert 0 < p.max_el_deg <= 90
        g = passes.geometry(els[METEOR_M2_3], STATION, np.array([p.aos, p.los]))
        np.testing.assert_allclose(g.el_deg, 0, atol=0.05)


def test_load_station(tmp_path: Path) -> None:
    a = tmp_path / "a.yaml"
    a.write_text("station_lat: 55.5\nstation_lon: 37.1\n")
    assert load_station(a) == Station(55.5, 37.1, 180.0, "+03:00")
    b = tmp_path / "b.yaml"
    b.write_text("lat: 1\nlon: 2\nalt_m: 3\ntz: '+00:00'\n")
    assert load_station(b) == Station(1.0, 2.0, 3.0, "+00:00")
    c0 = tmp_path / "c0.yaml"
    c0.write_text("station_lat: 55.5\nstation_lon: 37.1\naltitude_m: 172\n")
    assert load_station(c0).alt_m == 172.0
    c = tmp_path / "c.yaml"
    c.write_text("station_lat: 1\n")
    with pytest.raises(ValueError):
        load_station(c)
