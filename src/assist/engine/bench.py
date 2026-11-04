"""`assist bench` (plan §4.5, §8): render speed versus real time, on one worker and on N, without writing files.

- **One worker** (one core): this process renders the window after a 1 s warm-up (the chunk before the window:
  geometry blocks, passes, FFT plans), timing each stage: `control` (Python: per-emitter control samples and narrow
  samples), `synth`, `noise`, `adc` (kernel calls) and `prefetch` (serve only).
- **N workers**: a worker pool starts and renders the N * TASK_S seconds before the window (start-up and warm-up,
  timed apart), then renders the window: the steady file-render rate.

Both streams are hashed; they must be identical (the same bytes for any number of workers).
"""

import hashlib
import time
from typing import TYPE_CHECKING, Any

from assist.engine import workers as pool
from assist.engine.render import Renderer
from assist.engine.scene import Scene
from assist.sinks.runinfo import engine_name

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    import numpy as np

    from assist.config.resolve import Resolved


def _consume(parts: Iterable[np.ndarray]) -> tuple[float, str]:
    h = hashlib.blake2b(digest_size=16)
    tic = time.perf_counter()
    for u in parts:
        h.update(u)
    return time.perf_counter() - tic, h.hexdigest()


def run(r: Resolved, n0: int, n: int, workers: int, base: Path, log: Callable[[str], None]) -> dict[str, Any]:
    """Benchmark absolute samples [n0, n0 + n) of a resolved scenario (workers <= 1: one worker only)."""
    sc = r.scenario
    fs = int(sc.frontend.fs_hz)
    dur = n / fs
    tic = time.perf_counter()
    rd = Renderer(Scene(sc), base)
    rd.u8(n0 - fs, fs)
    warm = time.perf_counter() - tic
    rd.times = dict.fromkeys(rd.times, 0.0)
    log(f"1 worker: {dur:.1f} s of stream ({len(rd.emitters)} emitters) ...")
    t1, h1 = _consume(rd.iter_u8(n0, n))
    stages = {k: v / dur for k, v in rd.times.items()}
    one = {
        "seconds": round(t1, 3),
        "x_real_time": round(dur / t1, 2),
        "warm_up_s": round(warm, 3),
        "stage_s_per_stream_s": {k: round(v, 5) for k, v in stages.items()},
        "other_s_per_stream_s": round(t1 / dur - sum(stages.values()), 5),
    }
    out: dict[str, Any] = {
        "scenario_hash": r.hash,
        "engine": engine_name(rd),
        "start_sample_abs": n0,
        "duration_s": dur,
        "emitters": len(rd.emitters),
        "one_worker": one,
    }
    if workers > 1:
        nw = workers
        log(f"{nw} workers: start-up and {nw * pool.TASK_S} s warm-up ...")
        tic = time.perf_counter()
        with pool.RenderPool(r.canonical, base, nw, fs) as p:
            w0 = n0 - nw * pool.TASK_S * fs
            _consume(p.iter_u8(w0 - w0 % (pool.TASK_S * fs), nw * pool.TASK_S * fs))
            start = time.perf_counter() - tic
            log(f"{nw} workers: {dur:.1f} s of stream ...")
            tn, hn = _consume(p.iter_u8(n0, n))
        out["workers"] = {
            "count": nw,
            "seconds": round(tn, 3),
            "x_real_time": round(dur / tn, 2),
            "start_up_and_warm_up_s": round(start, 3),
            "identical_to_one_worker": hn == h1,
        }
    return out
