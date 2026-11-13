"""`lora`: SpaceBEE-style LoRa bursts, BW 125 kHz, an event signal.

Per burst (drawn from the schedule's keyed slot stream): spreading factor `sf` and payload length
`payload_bytes`, each a number or an inclusive [lo, hi] range. A burst is `preamble` up-chirps (value 0), two sync
up-chirps (sync word nibbles x 8), 2.25 down-chirps and the payload chirps. The payload symbol count and the
airtime follow Semtech's formula:

    n_pl = 8 + max(ceil((8 PL - 4 SF + 28 + 16 CRC - 20 IH) / (4 (SF - 2 DE))) (CR + 4), 0)
    airtime = (preamble + 4.25 + n_pl) T_sym,  T_sym = 2^SF / BW

with explicit header (IH = 0) and low-data-rate optimisation DE = 1 when T_sym > 16 ms (`ldro: auto`). Payload
values are random (from `random_bytes` keyed (source, emitter, "burst") with the burst's start in whole
microseconds as counter); the 8 header symbols, and all of them with DE = 1, use the reduced rate (multiples of
4), as a real modem's would. Not decodable: no whitening, Gray coding, interleaving or Hamming code.
"""

import math
from functools import cache
from typing import TYPE_CHECKING, Any, Literal, override

import numpy as np
from pydantic import Field, field_validator

from assist.signals.base import Signal, SignalParams, register_signal

if TYPE_CHECKING:
    from types import ModuleType

    from assist.emitters.schedules import Event
    from assist.signals.base import Keys

US = 1_000_000


def _range(v: int | tuple[int, int]) -> tuple[int, int]:
    return (v, v) if isinstance(v, int) else v


class LoraParams(SignalParams):
    bw: float = Field(125_000.0, gt=0)
    sf: int | tuple[int, int] = Field((7, 12), description="Spreading factor per burst: a number or [lo, hi].")
    cr: int = Field(1, ge=1, le=4, description="Coding rate 4/(4 + cr).")
    preamble: int = Field(8, ge=1)
    payload_bytes: int | tuple[int, int] = Field((10, 60), description="Payload length per burst: number or [lo, hi].")
    crc: bool = True
    ldro: Literal["auto"] | bool = "auto"
    sync_word: int = Field(0x12, ge=0, le=255)

    @field_validator("sf")
    @classmethod
    def _sf(cls, v: int | tuple[int, int]) -> int | tuple[int, int]:
        lo, hi = _range(v)
        if not 6 <= lo <= hi <= 12:
            raise ValueError("sf must be in 6 .. 12 (lo <= hi)")
        return v

    @field_validator("payload_bytes")
    @classmethod
    def _pl(cls, v: int | tuple[int, int]) -> int | tuple[int, int]:
        lo, hi = _range(v)
        if not 0 <= lo <= hi <= 255:
            raise ValueError("payload_bytes must be in 0 .. 255 (lo <= hi)")
        return v


def t_sym(sf: int, bw: float) -> float:
    return float(1 << sf) / bw


def low_rate(p: LoraParams, sf: int) -> bool:
    return t_sym(sf, p.bw) > 0.016 if p.ldro == "auto" else bool(p.ldro)


def payload_symbols(p: LoraParams, sf: int, payload_bytes: int) -> int:
    """Semtech: n_pl = 8 + max(ceil((8 PL - 4 SF + 28 + 16 CRC - 20 IH) / (4 (SF - 2 DE))) (CR + 4), 0)."""
    de = 1 if low_rate(p, sf) else 0
    num = 8 * payload_bytes - 4 * sf + 28 + 16 * int(p.crc)
    return 8 + max(math.ceil(num / (4 * (sf - 2 * de))) * (p.cr + 4), 0)


def airtime_s(p: LoraParams, sf: int, payload_bytes: int) -> float:
    return (p.preamble + 4.25 + payload_symbols(p, sf, payload_bytes)) * t_sym(sf, p.bw)


@cache
def _layout(preamble: int, n_pl: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Segment starts and lengths in symbols, and the index of the first payload segment."""
    starts: list[float] = []
    lens: list[float] = []
    t = 0.0
    for _ in range(preamble + 2 + 2):  # up-chirps, sync, two full down-chirps
        starts.append(t)
        lens.append(1.0)
        t += 1.0
    starts.append(t)
    lens.append(0.25)
    t += 0.25
    first = len(starts)
    for _ in range(n_pl):
        starts.append(t)
        lens.append(1.0)
        t += 1.0
    return np.array(starts), np.array(lens), first


@register_signal("lora")
class LoRa(Signal):
    Params = LoraParams
    mode = "event"

    def __init__(self, params: SignalParams, keys: Keys, dsp: ModuleType) -> None:
        super().__init__(params, keys, dsp)
        self._key = keys.key64("burst")

    @property
    def lp(self) -> LoraParams:
        p = self.p
        assert isinstance(p, LoraParams)
        return p

    @override
    def bandwidth_hz(self) -> float:
        return self.lp.bw

    @override
    def preferred_rate(self) -> int:
        return 100

    @override
    @classmethod
    def draw_event(cls, p: SignalParams, g: np.random.Generator) -> tuple[float, dict[str, Any]]:
        assert isinstance(p, LoraParams)
        sf = int(g.integers(_range(p.sf)[0], _range(p.sf)[1], endpoint=True))
        pl = int(g.integers(_range(p.payload_bytes)[0], _range(p.payload_bytes)[1], endpoint=True))
        d = airtime_s(p, sf, pl)
        return d, {"duration_s": d, "sf": sf, "payload_bytes": pl}

    @override
    @classmethod
    def max_duration_s(cls, p: SignalParams) -> float:
        assert isinstance(p, LoraParams)
        lo, hi = _range(p.sf)
        return max(airtime_s(p, sf, _range(p.payload_bytes)[1]) for sf in range(lo, hi + 1))

    def segments(self, ev: Event) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """(start s, length s, value) of each chirp segment of a burst (value -1: down-chirp), and its sf."""
        p = self.lp
        sf = int(ev.params["sf"])
        n_pl = payload_symbols(p, sf, int(ev.params["payload_bytes"]))
        starts, lens, first = _layout(p.preamble, n_pl)
        n = 2**sf
        val = np.zeros(len(starts), dtype=np.int32)
        val[p.preamble] = (p.sync_word >> 4) * 8 % n
        val[p.preamble + 1] = (p.sync_word & 0xF) * 8 % n
        val[p.preamble + 2 : first] = -1
        raw = self.dsp.random_bytes(self._key, round(ev.start * US), 2 * n_pl).view("<u2").astype(np.int64)
        vals = raw & (n - 1)
        reduced = np.zeros(n_pl, dtype=bool)
        reduced[:8] = True
        if low_rate(p, sf):
            reduced[:] = True
        vals[reduced] &= ~np.int64(3)  # reduced rate: multiples of 4
        val[first:] = vals.astype(np.int32)
        ts = t_sym(sf, p.bw)
        return starts * ts, lens * ts, val, sf

    @override
    def render_events(self, events: list[Event], n0: int, n: int, rate: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.complex64)
        for ev in events:
            st, ln, val, sf = self.segments(ev)
            s_us = round(ev.start * US)
            end_us = s_us + math.ceil((st[-1] + ln[-1]) * US)
            # sample j (index n0 + j) is at ((n0 + j) US - s_us rate) / (US rate) s after the burst start
            j0 = max(0, -((n0 * US - s_us * rate) // US))
            j1 = min(n, -((n0 * US - end_us * rate) // US))
            if j1 <= j0:
                continue
            t0 = ((n0 + j0) * US - s_us * rate) / (US * rate)
            out[j0:j1] += self.dsp.chirp(st, ln, val, sf, self.lp.bw, t0, 1.0 / rate, j1 - j0)
        return out
