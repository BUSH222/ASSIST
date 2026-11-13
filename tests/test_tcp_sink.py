"""The external raw TCP sink `tcp` (PLAN.md R3): formats, the shared server, and `assist run` end to end."""

import asyncio
import json
import logging
import socket
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from typer.testing import CliRunner, Result

from assist import plugins, registry
from assist.cli import app
from assist.sinks.base import Sink, check_sinks, register_sink, sink_class
from assist.sinks.stream import RenderAhead
from assist.timebase import parse_iso, sample_at

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = "examples/plugins/tcp_sink.py"
FS = 2_400_000
N0 = sample_at(parse_iso("2026-11-04T12:00:00Z"), FS)
runner = CliRunner()


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.chdir(ROOT)
    plugins.activate([{"path": PLUGIN}], (ROOT,))
    yield
    plugins.activate([])


def _mod() -> Any:
    return plugins.active()[0].module


class Ramp:
    """Stub source: sample n carries n mod 2^16 (I = low byte, Q = high byte)."""

    def iter_u8(self, n0: int, n: int) -> Iterator[np.ndarray]:
        idx = np.arange(n0, n0 + n, dtype=np.int64)
        out = np.empty(2 * n, dtype=np.uint8)
        out[0::2] = idx & 0xFF
        out[1::2] = (idx >> 8) & 0xFF
        yield out


def _ramp_bytes(n0: int, n: int) -> bytes:
    return next(Ramp().iter_u8(n0, n)).tobytes()


def _run(srv: Any, body: Any) -> Any:
    async def main() -> Any:
        task = asyncio.create_task(srv.serve())
        listening = asyncio.create_task(srv.listening.wait())
        await asyncio.wait({task, listening}, return_when=asyncio.FIRST_COMPLETED)
        if task.done():
            task.result()
        try:
            return await body()
        finally:
            srv.stop()
            await asyncio.wait_for(task, 10)

    return asyncio.run(main())


# --- registry and settings ---


def test_core_sinks_and_the_external_one() -> None:
    assert registry.sinks.names() == ["wav", "cu8", "fft", "rtl_tcp", "tcp"]
    assert registry.sinks.origin("tcp") == PLUGIN and registry.sinks.origin("rtl_tcp") == "core"
    assert [sink_class(n).serving for n in ("wav", "cu8", "fft", "rtl_tcp", "tcp")] == [False] * 3 + [True] * 2
    assert check_sinks([{"tcp": {"format": "cs16", "lookahead": "4s"}}, "cu8", {"rtl_tcp": None}]) == [
        {
            "tcp": {
                "host": "127.0.0.1",
                "port": 5555,
                "pacing": "realtime",
                "lookahead": 4.0,
                "format": "cs16",
                "max_backlog_s": 10.0,
            }
        },
        {"cu8": {}},
        {"rtl_tcp": {"host": "127.0.0.1", "port": 1234, "pacing": "realtime", "lookahead": 1.0}},
    ]
    with pytest.raises(ValueError, match="format"):
        check_sinks([{"tcp": {"format": "cs8"}}])
    plugins.activate([])
    with pytest.raises(ValueError, match=r"unknown sink 'tcp' \(known: wav, cu8, fft, rtl_tcp\).*plugins:"):
        check_sinks([{"tcp": {}}])


def test_register_sink_checks_the_class() -> None:
    with pytest.raises(TypeError, match="Sink subclass"):
        register_sink("x")(int)  # ty: ignore[invalid-argument-type]
    bad: Any = type("Serving", (Sink,), {"serving": True, "__module__": "some_plugin"})
    with pytest.raises(TypeError, match="derives from ServingSink"):
        register_sink("x")(bad)


def test_formats() -> None:
    enc = _mod().encode
    raw = bytes([0, 255, 127, 128])
    assert enc(raw, "cu8") == raw
    assert np.frombuffer(enc(raw, "cs16"), "<i2").tolist() == [-32640, 32640, -128, 128]  # 256 (u - 127.5)
    np.testing.assert_allclose(np.frombuffer(enc(raw, "cf32"), "<f4"), [-1.0, 1.0, -1 / 255, 1 / 255], rtol=1e-6)


# --- the server ---


@pytest.mark.parametrize("fmt", ["cu8", "cs16", "cf32"])
def test_stream_has_no_header_and_is_the_encoded_render(fmt: str) -> None:
    n = 100_001
    mod = _mod()
    srv = mod.TcpServer(RenderAhead(Ramp(), N0, n, 16, 240_000), FS, 137e6, fmt, port=0, pacing="unpaced")
    assert srv.bytes_per_sample == {"cu8": 2, "cs16": 4, "cf32": 8}[fmt]

    async def body() -> bytes:
        r, w = await asyncio.open_connection("127.0.0.1", srv.port)
        w.write(b"\x01\x00\x00\x00\x00 anything a client sends is ignored")
        data = await r.read(-1)  # to EOF: the stream ends after n samples
        w.close()
        return data

    data = _run(srv, body)
    assert data == mod.encode(_ramp_bytes(N0, n), fmt)  # from the first byte: no header
    assert srv.stats.samples == n and srv.stats.clients == 1


def test_slow_client_is_dropped(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="assist.serve")
    ahead = RenderAhead(Ramp(), N0, None, 16, 240_000)
    srv = _mod().TcpServer(ahead, FS, 137e6, "cf32", port=0, max_backlog_s=0.1, status_s=3600.0)
    assert srv.max_backlog_bytes == round(8 * 0.1 * FS)  # the limit is in seconds of stream, whatever the format

    async def body() -> None:
        _r, w = await asyncio.open_connection("127.0.0.1", srv.port)  # never reads
        t_end = time.monotonic() + 15
        while srv.stats.slow_disconnects == 0 and time.monotonic() < t_end:
            await asyncio.sleep(0.01)
        w.close()

    _run(srv, body)
    assert srv.stats.slow_disconnects == 1
    assert any("behind and was disconnected" in m for m in caplog.messages)
    assert any(m.startswith("tcp listening on") for m in caplog.messages)


# --- assist run ---


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_run_serves_the_bytes_of_a_file_render(tmp_path: Path) -> None:
    port = _free_port()
    sc = tmp_path / "s.yaml"
    sc.write_text(
        "version: 1\nseed: 3\n"
        'clock: {start: "2026-11-04T12:00:00Z", duration: 1s}\n'
        "station: {file: null, lat: 55.5, lon: 37.1, alt_m: 172}\n"
        "profile: profiles/vhf137\n"
        f"plugins: [{{path: {PLUGIN}}}]\n"
        f"sinks: [{{tcp: {{port: {port}, format: cs16, pacing: unpaced}}}}, {{cu8: {{}}}}]\n"
        "sources:\n"
        "  - id: z1\n    preset: zombie\n"
        "    orbit: {synthetic_pass: {aos: -5min, max_el_deg: 60, direction: desc, side: east}}\n"
        "    freq_hz: 137.35e6\n    schedule: {type: continuous}\n    level: {cn0_zenith_dbhz: 60}\n"
    )
    mixed = runner.invoke(app, ["run", str(sc), "-o", str(tmp_path / "x")])
    assert mixed.exit_code == 2 and "tcp (serving) and cu8 (files)" in mixed.output
    result: dict[str, Result] = {}

    def serve() -> None:
        args = ["run", str(sc), "--sink", "tcp", "--duration", "0.5s", "-o", str(tmp_path / "srv")]
        result["r"] = runner.invoke(app, args)

    th = threading.Thread(target=serve)
    th.start()
    data = b""
    t_end = time.monotonic() + 30
    while time.monotonic() < t_end:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=10) as s:
                chunks = []
                while got := s.recv(1 << 20):
                    chunks.append(got)
                data = b"".join(chunks)
            break
        except ConnectionRefusedError:
            time.sleep(0.05)
    th.join(30)
    assert result["r"].exit_code == 0, result["r"].output
    rr = runner.invoke(app, ["run", str(sc), "--sink", "cu8", "-o", str(tmp_path / "ren"), "--duration", "0.5s"])
    assert rr.exit_code == 0, rr.output
    (cu8,) = (tmp_path / "ren").glob("*.cu8")
    assert len(data) == 4 * (FS // 2) and data == _mod().encode(cu8.read_bytes(), "cs16")
    info = json.loads((tmp_path / "srv" / "run.json").read_text())
    (sink,) = info["sinks"]
    assert info["status"] == "complete" and sink["kind"] == "tcp" and sink["params"]["format"] == "cs16"
    assert sink["info"]["format"] == "cs16" and sink["info"]["port"] == port and sink["info"]["live"] is False
    assert sink["info"]["stream"] == {
        "samples": FS // 2,
        "seconds": 0.5,
        "stalls": 0,
        "stall_s": 0.0,
        "clients": 1,
        "slow_disconnects": 0,
    }
    assert info["plugins"][0]["plugin"] == PLUGIN and info["plugins"][0]["version"] == "0.1.0"
    assert len(info["plugins"][0]["sha256"]) == 64
    assert info["api_version"] == "1.0" and info["plugins"][0]["api"] == "1.0"
    assert info["plugins"][0]["file"] == str(ROOT / PLUGIN) and info["plugins"][0]["distribution"] is None
    assert info["plugins"][0]["registers"] == ["sink tcp"]
    file_run = json.loads((tmp_path / "ren" / "run.json").read_text())
    assert file_run["scenario_hash"] == info["scenario_hash"]
