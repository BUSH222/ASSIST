"""Serving the uint8 IQ stream over TCP: the parts every serving sink shares (PLAN.md R3).

- **Render-ahead (`RenderAhead`):** one thread renders the stream into a ring of about `lookahead` seconds, in chunks
  of 16 384 samples. The stream starts once the ring is full (pre-roll).
- **Fan-out (`StreamServer`):** every client gets the same chunks, from the first one sent after it connects. With
  nobody connected the chunks are dropped and the stream keeps running, so its sample time keeps tracking wall time.
- **Pacing (`realtime`):** chunk k is due at anchor + k * 16384 / fs on the monotonic clock. If the ring is empty at a
  due time, the stream stalls: the stall is logged and the anchor moves by its length, so content is unchanged and
  only the wall-clock alignment slips. With a live start (`wall_start`) the anchor is the wall time of the first
  sample, so sample time equals wall time.
- **`unpaced`:** chunks go out as fast as the slowest client reads; the stream holds while nobody is connected.
- **Backlog guard:** no per-client dropping of samples. A client whose unsent data grows past `max_backlog_s` of
  stream is disconnected, so a stuck client cannot exhaust memory.

A protocol is a `StreamServer` subclass: what it sends on connect (`greeting`), what it does with what clients send
(`read_client`), and how a chunk goes on the wire (`encode`). `StreamSink` makes a sink of one: `rtl_tcp` in
`rtltcp.py` is the core one.
"""

import asyncio
import contextlib
import logging
import math
import queue
import sys
import threading
import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, override

from assist.sinks.base import ServingSink
from assist.timebase import format_iso

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np

log = logging.getLogger("assist.serve")

CHUNK_SAMPLES = 16_384
PACINGS = ("realtime", "unpaced")


def live_start(now_unix: float, lookahead_s: float, margin_s: float = 1.0) -> int:
    """The whole Unix second a live stream starts at: after the pre-roll has had time to render."""
    return math.ceil(now_unix + lookahead_s + margin_s)


def stderr_handler() -> logging.Handler:
    """The serve log on stderr, with UTC wall-clock timestamps."""
    h = logging.StreamHandler(sys.stderr)
    f = logging.Formatter("%(asctime)s.%(msecs)03dZ %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
    f.converter = time.gmtime
    h.setFormatter(f)
    return h


class U8Source(Protocol):
    """Anything that renders the uint8 stream by absolute sample index (`engine.render.Renderer`)."""

    def iter_u8(self, n0: int, n: int) -> Iterable[np.ndarray]: ...


class RenderAhead:
    """A thread rendering absolute samples [n0, n0 + n) (n None: endless) into a bounded queue of byte chunks.

    Pieces are rendered `piece` samples at a time on the absolute grid (hop-aligned for the Renderer), then cut into
    chunks of `chunk` samples counted from n0. The queue ends with None; a render error is re-raised by `get`.
    """

    def __init__(
        self, source: U8Source, n0: int, n: int | None, capacity: int, piece: int, chunk: int = CHUNK_SAMPLES
    ) -> None:
        self.source, self.n0, self.n, self.piece, self.chunk = source, n0, n, piece, chunk
        self.capacity = capacity
        self._q: queue.Queue[bytes | None] = queue.Queue(maxsize=capacity)
        self._stop = threading.Event()
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="assist-render-ahead", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    @property
    def fill(self) -> int:
        """Chunks waiting in the ring."""
        return self._q.qsize()

    @property
    def ready(self) -> bool:
        """The ring is full, or the renderer has finished (short streams, errors)."""
        return self._q.full() or not self._thread.is_alive()

    def _put(self, item: bytes | None) -> bool:
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        try:
            cb = 2 * self.chunk
            buf = bytearray()
            n = self.n0
            end = None if self.n is None else self.n0 + self.n
            while end is None or n < end:
                hi = (n // self.piece + 1) * self.piece
                if end is not None:
                    hi = min(hi, end)
                for u8 in self.source.iter_u8(n, hi - n):
                    buf += u8.tobytes()
                n = hi
                while len(buf) >= cb:
                    if not self._put(bytes(buf[:cb])):
                        return
                    del buf[:cb]
            if buf:
                self._put(bytes(buf))
        except BaseException as e:  # handed to the consumer
            self.error = e
        finally:
            self._put(None)

    def _checked(self, item: bytes | None) -> bytes | None:
        if item is None and self.error is not None:
            raise self.error
        return item

    def get_nowait(self) -> bytes | None:
        """The next chunk, or None at the end; raises queue.Empty when the ring is empty."""
        return self._checked(self._q.get_nowait())


@dataclass
class _Client:
    peer: str
    writer: asyncio.StreamWriter
    since: int | None = None  # absolute sample of the first chunk sent to it
    sent: int = 0  # samples


@dataclass
class Stats:
    samples: int = 0  # samples streamed (sent or dropped)
    stalls: int = 0
    stall_s: float = 0.0
    clients: int = 0  # connections accepted
    slow_disconnects: int = 0
    min_fill: int | None = None  # lowest ring fill seen at a due time (realtime)


class StreamServer:
    """Accepts clients and streams a `RenderAhead` to all of them; subclasses add a protocol (see the module text)."""

    label = "tcp"  # the protocol's name in log lines
    bytes_per_sample = 2  # of the encoded stream (uint8 I, Q)

    def __init__(
        self,
        ahead: RenderAhead,
        fs: int,
        fc_hz: float,
        *,
        host: str = "127.0.0.1",
        port: int = 1234,
        pacing: str = "realtime",
        wall_start: float | None = None,
        max_backlog_s: float = 10.0,
        status_s: float = 60.0,
    ) -> None:
        if pacing not in PACINGS:
            raise ValueError(f"pacing must be one of {PACINGS}, got {pacing!r}")
        if wall_start is not None and pacing != "realtime":
            raise ValueError("a live start needs realtime pacing")
        self.ahead, self.fs, self.fc_hz = ahead, fs, fc_hz
        self.host, self.port, self.pacing = host, port, pacing
        self.wall_start = wall_start
        self.max_backlog_bytes = round(self.bytes_per_sample * max_backlog_s * fs)
        self.status_s = status_s
        self.n_next = ahead.n0  # absolute index of the next sample to stream
        self.stats = Stats()
        self.listening = asyncio.Event()
        self.started = asyncio.Event()  # the first chunk is due (after the pre-roll)
        self._stop = asyncio.Event()
        self._clients: list[_Client] = []

    def _at(self) -> str:
        return f"sample {self.n_next} ({format_iso(self.n_next / self.fs, 3)})"

    def stop(self) -> None:
        self._stop.set()

    def clients(self) -> list[tuple[str, int | None]]:
        """(peer, absolute sample of its first chunk, None before one was sent) of each connected client."""
        return [(c.peer, c.since) for c in self._clients]

    async def serve(self) -> None:
        """Listen, stream until the end of the render or `stop()`, then close every connection."""
        srv = await asyncio.start_server(self._handle, self.host, self.port)
        self.port = srv.sockets[0].getsockname()[1]
        fc = round(self.fc_hz)
        log.info(
            "%s listening on %s:%d (%s, %d S/s at %d Hz)", self.label, self.host, self.port, self.pacing, self.fs, fc
        )
        self.listening.set()
        self.ahead.start()
        try:
            await self._stream()
        finally:
            srv.close()
            for c in list(self._clients):
                self._drop(c, "closed by the server")
            self.ahead.stop()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(srv.wait_closed(), 2)

    # --- clients ---

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        c = _Client(f"{peer[0]}:{peer[1]}" if isinstance(peer, tuple) else str(peer), writer)
        hello = self.greeting()
        if hello:
            writer.write(hello)
        self._clients.append(c)
        self.stats.clients += 1
        log.info("client %s connected at %s", c.peer, self._at())
        try:
            await self.read_client(reader, c.peer)
        except asyncio.IncompleteReadError, ConnectionError:
            pass
        finally:
            self._drop(c, "disconnected")

    # --- the protocol (overridden by subclasses) ---

    def greeting(self) -> bytes:
        """Sent to each client when it connects (default: nothing)."""
        return b""

    async def read_client(self, reader: asyncio.StreamReader, peer: str) -> None:
        """Consume what a client sends until it disconnects (default: discard it)."""
        while await reader.read(4096):
            pass

    def encode(self, chunk: bytes) -> bytes:
        """A chunk of interleaved uint8 I, Q as it goes on the wire (default: unchanged)."""
        return chunk

    def _drop(self, c: _Client, why: str) -> None:
        if c not in self._clients:
            return
        self._clients.remove(c)
        c.writer.close()
        log.info("client %s %s at %s after %.1f s of stream", c.peer, why, self._at(), c.sent / self.fs)

    async def _send(self, chunk: bytes) -> None:
        n = len(chunk) // 2
        data = self.encode(chunk) if self._clients else chunk
        for c in list(self._clients):
            if c.writer.is_closing():
                self._drop(c, "disconnected")
                continue
            if c.since is None:
                c.since = self.n_next
            c.writer.write(data)
            c.sent += n
            backlog = c.writer.transport.get_write_buffer_size()
            if backlog > self.max_backlog_bytes:
                self.stats.slow_disconnects += 1
                behind = backlog / self.bytes_per_sample / self.fs
                self._drop(c, f"fell {behind:.1f} s behind and was disconnected")
        if self.pacing == "unpaced" and self._clients:
            await asyncio.gather(*(c.writer.drain() for c in self._clients), return_exceptions=True)
        self.n_next += n
        self.stats.samples += n

    # --- the stream ---

    async def _preroll(self) -> None:
        tic = time.monotonic()
        while not self.ahead.ready and not self._stop.is_set():
            if self.wall_start is not None and time.time() >= self.wall_start:
                log.warning(
                    "pre-roll incomplete at the live start (%d of %d chunks)", self.ahead.fill, self.ahead.capacity
                )
                return
            await asyncio.sleep(0.01)
        rendered = self.ahead.fill * self.ahead.chunk / self.fs
        log.info("pre-roll: %.2f s rendered in %.2f s", rendered, time.monotonic() - tic)

    async def _stream(self) -> None:
        await self._preroll()
        chunk_s = self.ahead.chunk / self.fs
        anchor = time.monotonic()
        if self.wall_start is not None:
            anchor += self.wall_start - time.time()
        log.info("streaming from %s%s", self._at(), " (live: each sample at its wall time)" if self.wall_start else "")
        self.started.set()
        next_status = anchor + self.status_s
        k = 0
        while not self._stop.is_set():
            if self.pacing == "realtime":
                delay = anchor + k * chunk_s - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                fill = self.ahead.fill
                self.stats.min_fill = fill if self.stats.min_fill is None else min(self.stats.min_fill, fill)
            else:
                while not self._clients and not self._stop.is_set():
                    await asyncio.sleep(0.01)  # unpaced: hold while nobody is connected
            try:
                chunk = self.ahead.get_nowait()
            except queue.Empty:
                if self.pacing == "realtime":
                    chunk = await self._stall()
                    anchor = time.monotonic() - k * chunk_s
                else:
                    chunk = await self._wait()
            if chunk is None:
                log.info("%s at %s", "stopped" if self._stop.is_set() else "end of stream", self._at())
                return
            await self._send(chunk)
            k += 1
            if self.pacing == "realtime" and time.monotonic() >= next_status:
                self._status()
                next_status += self.status_s

    async def _wait(self) -> bytes | None:
        """The next chunk, waiting for the renderer (polled); None at the end or on stop()."""
        while not self._stop.is_set():
            try:
                return self.ahead.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.001)
        return None

    async def _stall(self) -> bytes | None:
        tic = time.monotonic()
        log.warning("stall at %s: the renderer is behind", self._at())
        chunk = await self._wait()
        d = time.monotonic() - tic
        self.stats.stalls += 1
        self.stats.stall_s += d
        log.warning("stall over after %.3f s (wall-clock slip so far %.3f s)", d, self.stats.stall_s)
        return chunk

    def _status(self) -> None:
        log.info(
            "status: %s, %d client(s), ring %d/%d (min %s since the last status), %d stall(s)",
            self._at(),
            len(self._clients),
            self.ahead.fill,
            self.ahead.capacity,
            self.stats.min_fill,
            self.stats.stalls,
        )
        self.stats.min_fill = None


class StreamSink(ServingSink):
    """A serving sink made of a `StreamServer`: subclasses say which (`make_server`)."""

    server: StreamServer | None = None

    @abstractmethod
    def make_server(self, ahead: RenderAhead) -> StreamServer:
        """The server for this run (host, port, pacing from `self.sp`; the live start from `self.ctx.wall_start`)."""

    def server_args(self) -> dict[str, Any]:
        """Keyword arguments every `StreamServer` takes, from the sink's settings."""
        return {
            "host": self.sp.host,
            "port": self.sp.port,
            "pacing": self.sp.pacing,
            "wall_start": self.ctx.wall_start,
        }

    @override
    async def serve(self, ahead: RenderAhead) -> None:
        self.server = self.make_server(ahead)
        await self.server.serve()

    @override
    def info(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "host": self.sp.host,
            "port": self.sp.port if self.server is None else self.server.port,
            "pacing": self.sp.pacing,
            "live": self.ctx.wall_start is not None,
            "lookahead_s": self.sp.lookahead,
        }
        if self.server is not None:
            st = self.server.stats
            out["stream"] = {
                "samples": st.samples,
                "seconds": st.samples / self.ctx.fs,
                "stalls": st.stalls,
                "stall_s": round(st.stall_s, 3),
                "clients": st.clients,
                "slow_disconnects": st.slow_disconnects,
            }
        return out

    @override
    def summary(self) -> str | None:
        if self.server is None:
            return None
        st = self.server.stats
        return f"{self.kind}: {st.samples / self.ctx.fs:.1f} s streamed, {st.stalls} stall(s), {st.clients} client(s)"
