"""`tcp`: the IQ stream over a plain TCP socket, as cu8, cs16 or cf32. No header, no commands.

An external ASSIST sink: this file is everything it needs. A scenario uses it by listing the file:

    plugins:
      - path: examples/plugins/tcp_sink.py
    sinks:
      - tcp: {port: 5555, format: cs16}

and `assist run scenario.yaml` (or `--sink tcp` when the scenario lists other sinks too) serves it:

    nc 127.0.0.1 5555 > capture.cs16

Each client gets the stream from the moment it connects. Formats (interleaved I, Q, little endian):

- `cu8`:  the stream's own bytes (0..255, 127.5 is zero), as rtl_sdr writes;
- `cs16`: 256 * (u - 127.5), i.e. the same value with eight more bits (exact);
- `cf32`: (u - 127.5) / 127.5 as float32, full scale +-1.

Everything a serving sink needs is shared: `api.StreamSink` / `api.StreamServer` render ahead in one thread, pace the
stream (`pacing: realtime` or `unpaced`), start it live when the scenario clock is `now`, fan it out to every client,
and disconnect a client that falls more than `max_backlog_s` of stream behind. This file only adds the settings and
how a chunk is put on the wire (`encode`).
"""

from typing import Any, Literal, override

import numpy as np
from pydantic import Field

from assist import api

__version__ = "0.1.0"
ASSIST_API = "1.0"  # the assist.api version this module was written for

BYTES_PER_SAMPLE = {"cu8": 2, "cs16": 4, "cf32": 8}


class TcpParams(api.ServingParams):
    # host, port, pacing and lookahead come from api.ServingParams.
    port: int = Field(5555, ge=0, le=65535, description="TCP port (0: any free port).")
    format: Literal["cu8", "cs16", "cf32"] = Field("cu8", description="Sample format on the wire.")
    max_backlog_s: float = Field(
        10.0, gt=0, description="A client this many seconds of stream behind is disconnected (slow-client drop)."
    )


def encode(chunk: bytes, fmt: str) -> bytes:
    """Interleaved uint8 I, Q as `fmt` (little endian)."""
    if fmt == "cu8":
        return chunk
    u = np.frombuffer(chunk, dtype=np.uint8)
    if fmt == "cs16":
        return ((2 * u.astype(np.int32) - 255) * 128).astype("<i2").tobytes()  # 256 (u - 127.5), in integers
    return ((u.astype(np.float32) - np.float32(127.5)) / np.float32(127.5)).astype("<f4").tobytes()


class TcpServer(api.StreamServer):
    label = "tcp"

    def __init__(self, ahead: api.RenderAhead, fs: int, fc_hz: float, fmt: str, **kw: Any) -> None:
        self.fmt = fmt
        self.bytes_per_sample = BYTES_PER_SAMPLE[fmt]  # before the base sizes the backlog limit in bytes
        super().__init__(ahead, fs, fc_hz, **kw)

    # No greeting, and whatever a client sends is discarded: the defaults of api.StreamServer.

    @override
    def encode(self, chunk: bytes) -> bytes:
        return encode(chunk, self.fmt)


@api.register_sink("tcp")
class TcpSink(api.StreamSink):
    Params = TcpParams

    @override
    def make_server(self, ahead: api.RenderAhead) -> TcpServer:
        p = self.p
        assert isinstance(p, TcpParams)
        return TcpServer(
            ahead, self.ctx.fs, self.ctx.fc_hz, p.format, max_backlog_s=p.max_backlog_s, **self.server_args()
        )

    @override
    def info(self) -> dict[str, Any]:
        p = self.p
        assert isinstance(p, TcpParams)
        return {**super().info(), "format": p.format, "max_backlog_s": p.max_backlog_s}
