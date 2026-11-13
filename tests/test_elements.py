"""Element store and `elements fetch`. No network: fetch is tested with a fake opener."""

import csv
import datetime as dt
import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from assist.orbits import elements as E

if TYPE_CHECKING:
    from collections.abc import Callable

FIXTURE = Path(__file__).parent / "fixtures" / "omm_2026-11-02.csv"
FM117 = 41188
T_REC = dt.datetime(2026, 11, 1, 16, 39, tzinfo=dt.UTC).timestamp()


def test_store_keeps_every_set_and_picks_nearest() -> None:
    st = E.ElementStore.load([FIXTURE])
    sets = st.sets(FM117)
    assert [s.omm["EPOCH"] for s in sets] == ["2026-11-02T02:37:16.867200", "2026-11-02T10:50:20.095872"]
    assert st.nearest(FM117, T_REC).omm["EPOCH"] == "2026-11-02T02:37:16.867200"
    assert st.nearest(FM117, T_REC + 5 * 86400).omm["EPOCH"] == "2026-11-02T10:50:20.095872"
    mid = (sets[0].epoch_unix + sets[1].epoch_unix) / 2
    assert st.nearest(FM117, mid) is sets[1]  # tie: the newer one
    with pytest.raises(KeyError):
        st.nearest(12345, T_REC)
    assert E.load_newest([FIXTURE])[FM117]["EPOCH"] == "2026-11-02T10:50:20.095872"  # the profile's rule


def test_same_epoch_in_two_files_is_kept_once(tmp_path: Path) -> None:
    copy_ = tmp_path / "copy.csv"
    copy_.write_text(FIXTURE.read_text())
    assert len(E.ElementStore.load([FIXTURE, copy_]).sets(FM117)) == 2


def test_json_and_tle_formats(tmp_path: Path) -> None:
    rows = E.read_omm_csv(FIXTURE)
    j = tmp_path / "e.json"
    j.write_text(json.dumps(rows))
    tle = tmp_path / "e.tle"
    from sgp4 import exporter

    from assist.orbits.catalog import _satrec

    lines = []
    for r in rows:
        l1, l2 = exporter.export_tle(_satrec(r))
        lines += [r["OBJECT_NAME"], l1, l2]
    tle.write_text("\n".join(lines) + "\n")
    a, b, c = (E.ElementStore.load([p]) for p in (FIXTURE, j, tle))
    assert a.norads() == b.norads() == c.norads()
    sa, sc = a.nearest(FM117, T_REC), c.nearest(FM117, T_REC)
    assert sc.name == "ORBCOMM FM117"
    assert abs(sa.epoch_unix - sc.epoch_unix) < 1e-3
    ga = E.satellite_from_omm(sa.omm).at(E.timescale().utc(2026, 11, 1, 17)).position.km
    gc = E.satellite_from_omm(sc.omm).at(E.timescale().utc(2026, 11, 1, 17)).position.km
    assert max(abs(x - y) for x, y in zip(ga, gc, strict=True)) < 0.01  # TLE precision
    with pytest.raises(ValueError, match="unknown element format"):
        E.read_elements(tmp_path / "e.xml")


def test_staleness_warning_only_for_old_epochs() -> None:
    s = E.ElementStore.load([FIXTURE]).nearest(FM117, T_REC)
    assert E.staleness_warning(s, s.epoch_unix + 13 * 86400) is None
    assert "days older" in str(E.staleness_warning(s, s.epoch_unix + 15 * 86400))
    # Epochs after the window (back-propagation, predictive sets such as CXO/XMM) never warn
    assert E.staleness_warning(s, s.epoch_unix - 60 * 86400) is None


class _Resp(io.BytesIO):
    pass


def _opener(body: bytes) -> tuple[Callable[[str], _Resp], list[str]]:
    calls: list[str] = []

    def op(url: str) -> _Resp:
        calls.append(url)
        return _Resp(body)

    return op, calls


def test_fetch_writes_dated_snapshots_and_never_overwrites(tmp_path: Path) -> None:
    body = FIXTURE.read_bytes()
    op, calls = _opener(body)
    now = dt.datetime(2026, 11, 3, 23, 30, tzinfo=dt.UTC)
    paths = E.fetch(["active", "satnogs"], tmp_path / "elements", now, op)
    assert [p.name for p in paths] == ["active_2026-11-03.csv", "satnogs_2026-11-03.csv"]
    assert calls == [E.FEEDS["active"], E.FEEDS["satnogs"]]
    assert all(p.read_bytes() == body for p in paths)
    assert not list((tmp_path / "elements").glob("*.part"))
    with pytest.raises(FileExistsError):
        E.fetch(["satnogs"], tmp_path / "elements", now, op)
    assert paths[1].read_bytes() == body


def test_fetch_rejects_bad_content(tmp_path: Path) -> None:
    rows = list(csv.reader(FIXTURE.open()))
    bad = "\n".join(",".join(r[1:]) for r in rows).encode()
    op, _ = _opener(bad)
    with pytest.raises(ValueError, match="OBJECT_NAME"):
        E.fetch(["active"], tmp_path, dt.datetime(2026, 11, 3, tzinfo=dt.UTC), op)
    assert not list(tmp_path.glob("active_*"))
    with pytest.raises(ValueError, match="unknown feed"):
        E.fetch(["celestrak"], tmp_path)


def _scenario_doc(elements: dict[str, object]) -> dict[str, Any]:
    from assist.config import loader

    doc = loader.load_scenario(Path(__file__).parent / "fixtures" / "scenario_basic.yaml")
    doc["elements"] = elements
    return doc


def test_latest_downloads_only_what_is_missing(tmp_path: Path) -> None:
    body = FIXTURE.read_bytes()
    op, calls = _opener(body)
    now = dt.datetime(2026, 11, 5, 8, tzinfo=dt.UTC)
    (tmp_path / "active_2026-11-05.csv").write_bytes(body)
    paths, warns = E.latest_snapshots(tmp_path, download=True, now=now, opener=op)
    assert calls == [E.FEEDS["satnogs"]] and not warns
    assert [p.name for p in paths] == ["active_2026-11-05.csv", "satnogs_2026-11-05.csv"]
    calls.clear()
    E.latest_snapshots(tmp_path, download=True, now=now, opener=op)
    assert calls == []  # today's files are there: no network


def test_latest_falls_back_to_existing_when_the_download_fails(tmp_path: Path) -> None:
    (tmp_path / "active_2026-11-02.csv").write_bytes(FIXTURE.read_bytes())

    def down(url: str) -> _Resp:
        raise OSError("offline")

    paths, warns = E.latest_snapshots(tmp_path, True, dt.datetime(2026, 11, 5, tzinfo=dt.UTC), down)
    assert [p.name for p in paths] == ["active_2026-11-02.csv"] and "offline" in warns[0]
    with pytest.raises(FileNotFoundError, match="no element snapshots"):
        E.latest_snapshots(tmp_path / "none", True, dt.datetime(2026, 11, 5, tzinfo=dt.UTC), down)


def test_latest_existing_picks_the_newest_per_feed(tmp_path: Path) -> None:
    for n in (
        "active_2026-10-30.csv",
        "active_2026-11-02.csv",
        "satnogs_2026-11-01.csv",
        "active_2026-11-09.csv.part",
    ):
        (tmp_path / n).write_text("x")
    paths, warns = E.latest_snapshots(tmp_path, download=False)
    assert [p.name for p in paths] == ["active_2026-11-02.csv", "satnogs_2026-11-01.csv"] and not warns
    with pytest.raises(FileNotFoundError, match="fetch_omm"):
        E.latest_snapshots(tmp_path / "none", download=False)


def test_scenario_latest_existing_resolves_to_concrete_files(tmp_path: Path) -> None:
    from assist.config.resolve import resolve_document

    (tmp_path / "elements").mkdir()
    (tmp_path / "elements" / "active_2026-11-02.csv").write_bytes(FIXTURE.read_bytes())
    r = resolve_document(_scenario_doc({"latest_existing": True}), tmp_path)
    assert r.scenario.elements.files == ["elements/active_2026-11-02.csv"]
    assert not r.scenario.elements.latest_existing  # the resolved scenario never looks again


def test_scenario_errors_name_the_problem(tmp_path: Path) -> None:
    from assist.config.loader import ConfigError
    from assist.config.resolve import resolve_document

    with pytest.raises(ConfigError, match="no element snapshots"):
        resolve_document(_scenario_doc({"latest_existing": True}), tmp_path)
    with pytest.raises(ConfigError, match=r"NORAD 41188 is in none of the element files"):
        resolve_document(_scenario_doc({}), tmp_path)
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_document(_scenario_doc({"latest": True, "latest_existing": True}), tmp_path)
