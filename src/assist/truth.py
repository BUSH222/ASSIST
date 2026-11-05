"""`assist truth` (plan §8): what the scene contains, computed on demand from the resolved scenario.

Continuous emitters give one row per step while the platform is above the horizon; event emitters give an event
list. Values are printed with fixed precision so the text is the same on every machine: frequencies to 1 mHz (from
integer mHz, so f_rx = f_tx + doppler exactly), angles to 1e-6 deg, range to 1 mm, range rate to 0.1 mm/s, levels
to 0.001 dB.

Ground and local sources (M7) have no geometry: their rows leave norad, elevation, azimuth, range and range rate
empty, and Doppler is 0. Every event (a VDL burst, an AM transmission, a 138 MHz carrier, a hump episode) lists its
frequency with its own offset and its strength (rx_dbm, cn0_dbhz) with its own level; a USB-hump episode has no
single frequency or level (its strength is `amplitude_db` in params, its spacing `spacing_hz`), so those cells are
empty. `platforms` restricts the output to some platform kinds.
"""

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, TextIO

import numpy as np

from assist.emitters.link import kt_dbw_hz
from assist.timebase import format_iso

if TYPE_CHECKING:
    from assist.engine.scene import Scene

COLUMNS = (
    "t_utc",
    "source_id",
    "emitter_id",
    "signal",
    "norad",
    "in_catalog",
    "f_rest_hz",
    "f_tx_hz",
    "f_rx_hz",
    "doppler_hz",
    "el_deg",
    "az_deg",
    "range_km",
    "range_rate_mps",
    "rx_dbm",
    "cn0_dbhz",
    "active_fraction",
)
EVENT_COLUMNS = ("start", "end", "source_id", "emitter_id", "signal", "f_rx_hz", "rx_dbm", "cn0_dbhz", "params")
PLATFORMS = ("satellite", "ground", "local")
_TEXT = {"t_utc", "source_id", "emitter_id", "signal", "start", "end", "params"}


def fmt_mhz(v: int) -> str:
    """Integer millihertz as a decimal Hz string."""
    sign = "-" if v < 0 else ""
    a = abs(int(v))
    return f"{sign}{a // 1000}.{a % 1000:03d}"


def fmt(x: float, decimals: int) -> str:
    """Fixed precision; NaN (no geometry, no single level) prints as an empty cell."""
    if math.isnan(float(x)):
        return ""
    return f"{round(float(x), decimals) + 0.0:.{decimals}f}"  # + 0.0 turns -0.0 into 0.0


@dataclass
class TruthTable:
    rows: list[dict[str, str]]
    events: list[dict[str, str]]


def compute(
    scene: Scene, t0: float, t1: float, step_s: float = 1.0, platforms: tuple[str, ...] = PLATFORMS
) -> TruthTable:
    """Rows at t0 + k * step (< t1) for continuous emitters, and events overlapping [t0, t1), of sources whose
    platform is in `platforms`."""
    n = max(0, int(np.ceil((t1 - t0) / step_s - 1e-9)))
    t = t0 + np.arange(n) * step_s
    rows: list[tuple[float, str, str, dict[str, str]]] = []
    events: list[tuple[float, str, str, dict[str, str]]] = []
    for src in scene.sources:
        if src.config.platform not in platforms:
            continue
        plat = src.platform
        norad = "" if plat.norad is None else str(plat.norad)
        in_cat = "true" if plat.in_catalog else "false"
        for em in src.emitters:
            if em.is_event:
                for et in em.event_truth(t0, t1):
                    events.append(
                        (
                            et.event.start,
                            src.id,
                            em.id,
                            {
                                "start": format_iso(et.event.start, 3),
                                "end": format_iso(et.event.end, 3),
                                "source_id": src.id,
                                "emitter_id": em.id,
                                "signal": em.signal_kind,
                                "f_rx_hz": "" if em.signal_kind == "noise_comb" else fmt_mhz(et.f_rx_mhz),
                                "rx_dbm": fmt(et.rx_dbm, 3),
                                "cn0_dbhz": fmt(et.rx_dbm - 30 - kt_dbw_hz(em.t_sys_k), 3),
                                "params": json.dumps(
                                    {
                                        k: round(v, 6) if isinstance(v, float) else v
                                        for k, v in et.event.params.items()
                                    },
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            },
                        )
                    )
                continue
            tr = em.truth(t, step_s)
            f_rest = fmt_mhz(tr.f_rest_mhz)
            f_rx = tr.f_rx_mhz
            for i in range(tr.t.size):
                rows.append(
                    (
                        float(tr.t[i]),
                        src.id,
                        em.id,
                        {
                            "t_utc": format_iso(float(tr.t[i]), 3),
                            "source_id": src.id,
                            "emitter_id": em.id,
                            "signal": em.signal_kind,
                            "norad": norad,
                            "in_catalog": in_cat,
                            "f_rest_hz": f_rest,
                            "f_tx_hz": fmt_mhz(int(tr.f_tx_mhz[i])),
                            "f_rx_hz": fmt_mhz(int(f_rx[i])),
                            "doppler_hz": fmt_mhz(int(tr.doppler_mhz[i])),
                            "el_deg": fmt(tr.el_deg[i], 6),
                            "az_deg": fmt(tr.az_deg[i], 6),
                            "range_km": fmt(tr.range_m[i] / 1000.0, 6),
                            "range_rate_mps": fmt(tr.range_rate_m_s[i], 4),
                            "rx_dbm": fmt(tr.rx_dbm[i], 3),
                            "cn0_dbhz": fmt(tr.cn0_dbhz[i], 3),
                            "active_fraction": fmt(tr.active_fraction[i], 6),
                        },
                    )
                )
    rows.sort(key=lambda r: r[:3])
    events.sort(key=lambda r: r[:3])
    return TruthTable([r[3] for r in rows], [e[3] for e in events])


def _params_json(params: dict[str, object]) -> str:
    rounded = {k: round(v, 6) if isinstance(v, float) else v for k, v in params.items()}
    return json.dumps(rounded, sort_keys=True, separators=(",", ":"))


def write_csv(rows: list[dict[str, str]], columns: tuple[str, ...], f: TextIO) -> None:
    f.write(",".join(columns) + "\n")
    for r in rows:
        f.write(",".join(_csv_cell(r[c]) for c in columns) + "\n")


def _csv_cell(v: str) -> str:
    return '"' + v.replace('"', '""') + '"' if any(ch in v for ch in ',"\n') else v


def write_jsonl(rows: list[dict[str, str]], columns: tuple[str, ...], f: TextIO) -> None:
    """One JSON object per line; numbers keep the fixed-precision text of the CSV."""
    for r in rows:
        parts = []
        for c in columns:
            v = r[c]
            if c in _TEXT and c != "params":  # params is already JSON
                val = json.dumps(v)
            elif c == "norad":
                val = v or "null"
            else:
                val = v
            parts.append(f"{json.dumps(c)}:{val}")
        f.write("{" + ",".join(parts) + "}\n")
