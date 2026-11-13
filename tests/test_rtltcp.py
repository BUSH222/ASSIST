"""T-RTL (plan §10.1): rtl_tcp header, command parsing, fan-out, pacing; the rtl_tcp sink of `assist run`.

Most tests use a stub source whose sample n carries n mod 2^16 (I = low byte, Q = high byte), so continuity and
chunk alignment can be read from the bytes. `test_pacing_60s` runs a real minute of stream.
"""

import asyncio
import json
import logging
import math
import socket
import struct
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner, Result

from assist.cli import app
from assist.sinks.rtltcp import (
    CHUNK_SAMPLES,
    RenderAhead,
    RtlTcpServer,
    describe_command,
    dongle_info,
    live_start,
    parse_command,
)
from assist.timebase import parse_iso, sample_at

ROOT = Path(__file__).resolve().parent.parent
FS = 2_400_000
FC = 137e6
N0 = sample_at(parse_iso("2026-11-04T12:00:00Z"), FS)
PIECE = 240_000  # 10 hops, as the CLI uses
runner = CliRunner()


class Ramp:
    """Stub source: sample n carries n mod 2^16. `slow`: seconds of sleep per second of stream rendered."""

    def __init__(self, slow: float = 0.0) -> None:
        self.slow = slow

    def iter_u8(self, n0: int, n: int):
        if self.slow:
            time.sleep(self.slow * n / FS)
        idx = np.arange(n0, n0 + n, dtype=np.int64)
        out = np.empty(2 * n, dtype=np.uint8)
        out[0::2] = idx & 0xFF
        out[1::2] = (idx >> 8) & 0xFF
        yield out


def _index(b: bytes) -> np.ndarray:
    a = np.frombuffer(b, dtype=np.uint8).astype(np.int64)
    return a[0::2] | (a[1::2] << 8)


def _contiguous(b: bytes) -> bool:
    return bool(np.all((np.diff(_index(b)) - 1) % 65536 == 0))


def _server(source: Ramp | None = None, n0: int = N0, n: int | None = None, ring: int = 16, **kw) -> RtlTcpServer:
    ahead = RenderAhead(source or Ramp(), n0, n, ring, PIECE)
    kw.setdefault("status_s", 3600.0)
    return RtlTcpServer(ahead, FS, FC, port=0, **kw)


def _run(srv: RtlTcpServer, body):
    """Run the server, await body(), then stop the server and wait for it."""

    async def main():
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


async def _connect(srv: RtlTcpServer) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    r, w = await asyncio.open_connection("127.0.0.1", srv.port)
    assert await r.readexactly(12) == dongle_info()
    return r, w


async def _until(cond, timeout: float = 5.0) -> None:
    t_end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > t_end:
            raise TimeoutError("condition not met")
        await asyncio.sleep(0.01)


def test_dongle_info_is_r820t() -> None:
    assert dongle_info() == b"RTL0\x00\x00\x00\x05\x00\x00\x00\x1d"  # R820T, 29 gains


def test_parse_and_describe_commands() -> None:
    assert parse_command(struct.pack(">BI", 0x01, 137_000_000)) == (1, 137_000_000)
    assert parse_command(struct.pack(">Bi", 0x04, -10)) == (4, -10)  # gain, tenths of dB, signed
    assert parse_command(struct.pack(">Bi", 0x05, -3)) == (5, -3)  # ppm, signed
    assert parse_command(struct.pack(">BI", 0x0D, 0xFFFFFFFF)) == (13, 0xFFFFFFFF)  # index: unsigned
    assert describe_command(1, 137_000_000, FC, FS) == "set_frequency 137000000"
    assert describe_command(1, 137_100_000, FC, FS).endswith("(the stream is fixed at 137000000 Hz)")
    assert describe_command(2, 2_048_000, FC, FS).endswith("(the stream is fixed at 2400000 S/s)")
    assert describe_command(0x42, 7, FC, FS) == "unknown 0x42 7"


def test_live_start_is_a_whole_second_after_the_preroll() -> None:
    assert live_start(1000.2, 1.0) == 1003
    assert live_start(1000.0, 1.0, margin_s=0.5) == 1002


def test_header_stream_and_ignored_commands(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="assist.serve")
    srv = _server()

    async def body():
        r, w = await _connect(srv)
        first = await r.readexactly(2 * CHUNK_SAMPLES)
        cmd = struct.pack(">BI", 0x01, 137_100_000)
        w.write(cmd[:2])  # a command split across two TCP writes
        await w.drain()
        await asyncio.sleep(0.05)
        w.write(cmd[2:] + struct.pack(">Bi", 0x04, -10) + struct.pack(">BI", 0x02, 2_048_000))
        await w.drain()
        rest = await r.readexactly(2 * 20 * CHUNK_SAMPLES)
        await _until(lambda: sum("ignored at sample" in m for m in caplog.messages) >= 3)
        w.close()
        return first + rest

    data = _run(srv, body)
    assert _contiguous(data)
    cmds = [m for m in caplog.messages if "ignored at sample" in m]
    assert "set_frequency 137100000 (the stream is fixed at 137000000 Hz)" in cmds[0]
    assert "set_gain -10" in cmds[1] and "set_sample_rate 2048000" in cmds[2]


def test_fan_out_and_disconnect(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="assist.serve")
    srv = _server()

    async def body():
        ra, wa = await _connect(srv)
        a1 = await ra.readexactly(2 * 30 * CHUNK_SAMPLES)
        rb, wb = await _connect(srv)
        b = await rb.readexactly(2 * 30 * CHUNK_SAMPLES)
        sinces = dict(srv.clients())
        wb.close()
        await _until(lambda: len(srv.clients()) == 1)
        a2 = await ra.readexactly(2 * 30 * CHUNK_SAMPLES)
        wa.close()
        await _until(lambda: not srv.clients())  # else stop() may close A first ("closed by the server")
        return a1 + a2, b, sinces

    a, b, sinces = _run(srv, body)
    assert _contiguous(a) and _contiguous(b)
    for since in sinces.values():
        assert since is not None and (since - N0) % CHUNK_SAMPLES == 0  # clients join at chunk boundaries
        assert since % 65536 in (_index(a[:2])[0], _index(b[:2])[0])
    assert len(set(sinces.values())) == 2  # B joined later than A
    assert sum("disconnected at sample" in m for m in caplog.messages) == 2


def test_stream_runs_without_clients() -> None:
    """With nobody connected the stream keeps running in real time (chunks are dropped)."""
    srv = _server()

    async def body():
        await srv.started.wait()
        t0 = time.monotonic()
        await asyncio.sleep(0.5)
        r, w = await _connect(srv)
        await r.readexactly(2 * CHUNK_SAMPLES)
        ((_, since),) = srv.clients()
        w.close()
        return since, time.monotonic() - t0

    since, elapsed = _run(srv, body)
    assert since is not None
    assert abs((since - N0) / FS - elapsed) < 0.1


def test_stall_is_logged_and_content_unchanged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="assist.serve")
    srv = _server(Ramp(slow=2.0), ring=4)  # renders at half real time

    async def body():
        r, w = await _connect(srv)
        data = await r.readexactly(2 * round(0.3 * FS))
        w.close()
        return data

    data = _run(srv, body)
    assert _contiguous(data)
    assert srv.stats.stalls > 0 and srv.stats.stall_s > 0.1
    assert any("stall at sample" in m for m in caplog.messages)


def test_live_start_matches_wall_clock() -> None:
    wall = math.ceil(time.time() + 0.6)
    srv = _server(n0=wall * FS, wall_start=wall)

    async def body():
        r, w = await _connect(srv)
        await r.readexactly(1)
        t_first = time.time()
        await r.readexactly(2 * 10 * CHUNK_SAMPLES - 1)
        ((_, since),) = srv.clients()
        w.close()
        return t_first, since

    t_first, since = _run(srv, body)
    assert since == wall * FS  # connected before the start: the first sample is the live start
    assert abs(t_first - wall) < 0.05


def test_duration_ends_the_stream_unpaced() -> None:
    """Unpaced holds until a client connects, then sends as fast as it reads; --duration ends the stream."""
    n = 100_001
    srv = _server(n=n, pacing="unpaced")

    async def body():
        await asyncio.sleep(0.3)
        r, w = await _connect(srv)
        data = await r.read(-1)  # to EOF
        w.close()
        return data

    data = _run(srv, body)
    assert len(data) == 2 * n
    assert _contiguous(data) and _index(data[:2])[0] == N0 % 65536


def test_unpaced_runs_faster_than_real_time() -> None:
    srv = _server(pacing="unpaced", ring=64)

    async def body():
        r, w = await _connect(srv)
        t0 = time.monotonic()
        data = await r.readexactly(2 * 5 * FS)
        dt = time.monotonic() - t0
        w.close()
        return data, dt

    data, dt = _run(srv, body)
    assert _contiguous(data) and dt < 2.5


def test_stuck_client_is_disconnected(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="assist.serve")
    srv = _server(max_backlog_s=0.1)

    async def body():
        _r, w = await _connect(srv)  # never reads
        await _until(lambda: srv.stats.slow_disconnects == 1, timeout=15)
        w.close()

    _run(srv, body)
    assert any("behind and was disconnected" in m for m in caplog.messages)


def test_pacing_60s() -> None:
    """T-RTL: the stream rate over 60 s of real time is fs within 0.1 %, with no stalls."""
    srv = _server(ring=147)

    async def body():
        r, w = await _connect(srv)
        first = await r.read(1 << 20)
        t_first = time.monotonic()
        total = 0
        while True:
            got = await r.read(1 << 20)
            t_last = time.monotonic()
            total += len(got)
            if t_last - t_first >= 60:
                break
        w.close()
        return len(first), total, t_last - t_first

    first, total, elapsed = _run(srv, body)
    rate = total / 2 / elapsed
    assert abs(rate / FS - 1) < 1e-3, rate
    assert first > 0 and srv.stats.stalls == 0


# --- the CLI ---


@pytest.fixture
def _cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.mark.usefixtures("_cwd")
def test_serve_equals_render(tmp_path: Path) -> None:
    """The rtl_tcp sink streams exactly the bytes the cu8 sink writes for the same window; the port is the sink's."""
    port = _free_port()
    sc = tmp_path / "s.yaml"
    sc.write_text(
        "version: 1\nseed: 3\n"
        'clock: {start: "2026-11-04T12:00:00Z", duration: 1s}\n'
        "profile: profiles/vhf137\n"
        f"sinks: [{{rtl_tcp: {{port: {port}}}}}]\n"
        "sources:\n"
        "  - id: z1\n    preset: zombie\n"
        "    orbit: {synthetic_pass: {aos: -5min, max_el_deg: 60, direction: desc, side: east}}\n"
        "    freq_hz: 137.35e6\n    schedule: {type: continuous}\n    level: {cn0_zenith_dbhz: 60}\n"
    )
    result: dict[str, Result] = {}

    def run_serve() -> None:
        args = [
            "run",
            str(sc),
            "--set",
            "sinks[rtl_tcp].pacing=unpaced",
            "--duration",
            "0.5s",
            "-o",
            str(tmp_path / "srv"),
        ]
        result["r"] = runner.invoke(app, args)

    th = threading.Thread(target=run_serve)
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
    res = result["r"]
    assert res.exit_code == 0, res.output
    assert data[:12] == dongle_info()
    rr = runner.invoke(app, ["run", str(sc), "--sink", "cu8", "-o", str(tmp_path / "ren"), "--duration", "0.5s"])
    assert rr.exit_code == 0, rr.output
    (cu8,) = (tmp_path / "ren").glob("*.cu8")
    assert data[12:] == cu8.read_bytes()
    info = json.loads((tmp_path / "srv" / "run.json").read_text())
    (sink,) = info["sinks"]
    assert info["status"] == "complete" and sink["kind"] == "rtl_tcp" and info["files"] == []
    assert sink["params"] == {"host": "127.0.0.1", "port": port, "pacing": "unpaced", "lookahead": 1.0}
    stream = sink["info"].pop("stream")
    assert sink["info"] == {"host": "127.0.0.1", "port": port, "pacing": "unpaced", "live": False, "lookahead_s": 1.0}
    assert stream["samples"] == FS // 2 and stream["clients"] == 1 and stream["stalls"] == 0


@pytest.mark.usefixtures("_cwd")
def test_serve_rejects_bad_pacing() -> None:
    r = runner.invoke(app, ["run", "tests/fixtures/scenario_basic.yaml", "--set", "sinks[rtl_tcp].pacing=fast"])
    assert r.exit_code == 2 and "pacing" in r.output
