"""rtl_tcp server (plan §7.1): the station's uint8 IQ stream for read-only clients such as SDR++'s rtl_tcp source.

    sinks:
      - rtl_tcp: {host: 127.0.0.1, port: 1234, pacing: realtime, lookahead: 1s}

Protocol (librtlsdr's rtl_tcp): on connect the server sends 12 bytes of dongle info, "RTL0" + u32 BE tuner type
(5 = R820T) + u32 BE gain count (29), then interleaved uint8 I, Q. Clients may send 5-byte commands (u8 command,
u32 BE parameter); they are logged with the stream position and ignored, because the tuning is fixed.

Rendering ahead, fan-out, pacing and the backlog guard are shared by every serving sink: see `stream.py`.
"""

import struct
from typing import TYPE_CHECKING, override

from assist.sinks.base import ServingParams, register_sink
from assist.sinks.stream import (
    CHUNK_SAMPLES,
    PACINGS,
    RenderAhead,
    Stats,
    StreamServer,
    StreamSink,
    U8Source,
    live_start,
    log,
    stderr_handler,
)

if TYPE_CHECKING:
    import asyncio

__all__ = [
    "CHUNK_SAMPLES",
    "PACINGS",
    "RenderAhead",
    "RtlTcpServer",
    "RtlTcpSink",
    "Stats",
    "U8Source",
    "describe_command",
    "dongle_info",
    "live_start",
    "parse_command",
    "stderr_handler",
]

TUNER_R820T = 5
R820T_GAINS = 29
COMMANDS = {
    0x01: "set_frequency",
    0x02: "set_sample_rate",
    0x03: "set_gain_mode",
    0x04: "set_gain",
    0x05: "set_freq_correction",
    0x06: "set_if_gain",
    0x07: "set_test_mode",
    0x08: "set_agc_mode",
    0x09: "set_direct_sampling",
    0x0A: "set_offset_tuning",
    0x0B: "set_rtl_xtal",
    0x0C: "set_tuner_xtal",
    0x0D: "set_tuner_gain_by_index",
    0x0E: "set_bias_tee",
}
_SIGNED = {0x04, 0x05}  # gain in tenths of dB, frequency correction in ppm


def dongle_info(tuner: int = TUNER_R820T, gains: int = R820T_GAINS) -> bytes:
    return b"RTL0" + struct.pack(">II", tuner, gains)


def parse_command(b: bytes) -> tuple[int, int]:
    """(command, parameter) of one 5-byte command; the parameter is signed for gain and frequency correction."""
    cmd, param = struct.unpack(">BI", b)
    if cmd in _SIGNED and param >= 1 << 31:
        param -= 1 << 32
    return cmd, param


def describe_command(cmd: int, param: int, fc_hz: float, fs: int) -> str:
    text = f"{COMMANDS.get(cmd, f'unknown 0x{cmd:02x}')} {param}"
    if cmd == 0x01 and param != round(fc_hz):
        text += f" (the stream is fixed at {round(fc_hz)} Hz)"
    elif cmd == 0x02 and param != fs:
        text += f" (the stream is fixed at {fs} S/s)"
    return text


class RtlTcpServer(StreamServer):
    label = "rtl_tcp"

    @override
    def greeting(self) -> bytes:
        return dongle_info()

    @override
    async def read_client(self, reader: asyncio.StreamReader, peer: str) -> None:
        while True:
            cmd, param = parse_command(await reader.readexactly(5))
            what = describe_command(cmd, param, self.fc_hz, self.fs)
            log.info("client %s: %s, ignored at %s", peer, what, self._at())


@register_sink("rtl_tcp")
class RtlTcpSink(StreamSink):
    Params = ServingParams

    @override
    def make_server(self, ahead: RenderAhead) -> RtlTcpServer:
        return RtlTcpServer(ahead, self.ctx.fs, self.ctx.fc_hz, **self.server_args())
