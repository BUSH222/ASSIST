"""Chunk-parallel file rendering: worker processes, each with its own Renderer.

The window is cut at absolute multiples of TASK_S seconds; the tasks run on a pool of spawned processes and their
bytes come back in order, at most two tasks per worker in flight. Every chunk is a pure function of the resolved
scenario and its absolute index (a task's first chunk computes the block it overlaps from the previous chunk on its
own), so the stream is the same bytes for any number of workers. Processes rather than threads: the per-chunk
control plane (geometry, link budget, narrow samples) is Python and holds the GIL.
"""

import itertools
import json
import multiprocessing as mp
import signal
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from assist import plugins

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np

    from assist.engine.render import Renderer

TASK_S = 4  # seconds of stream per task (4 chunks: the lead-in block costs ~1 %)

_renderer: Renderer | None = None


def _init(canonical: str, base: str, located: dict[tuple[str, str], str]) -> None:
    """Worker start-up: build the Renderer from the resolved scenario (canonical JSON). The scenario's plugins are
    loaded first (the schema needs their kinds), from the files the parent process found them in."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C is the parent's to handle
    from assist.config import schema as S
    from assist.engine.render import Renderer
    from assist.engine.scene import Scene

    global _renderer
    plugins.remember(located)
    plugins.activate(json.loads(canonical).get("plugins"), (Path(base),))
    _renderer = Renderer(Scene(S.Scenario.model_validate_json(canonical)), Path(base))


def _task(n0: int, n: int) -> np.ndarray:
    if _renderer is None:
        raise RuntimeError("worker not initialised")
    return _renderer.u8(n0, n)


def tasks(n0: int, n: int, fs: int, task_s: int = TASK_S) -> list[tuple[int, int]]:
    """[n0, n0 + n) cut at absolute multiples of task_s seconds: (first sample, count) per task."""
    size = task_s * fs
    out = []
    a, end = n0, n0 + n
    while a < end:
        b = min(end, (a // size + 1) * size)
        out.append((a, b - a))
        a = b
    return out


class RenderPool:
    """Worker processes rendering one resolved scenario; `iter_u8` can be called more than once (bench warms up)."""

    def __init__(self, canonical: str, base: Path, workers: int, fs: int, task_s: int = TASK_S) -> None:
        self.workers, self.fs, self.task_s = max(1, workers), fs, task_s
        self._ex = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init,
            initargs=(canonical, str(base), plugins.located()),
        )

    def iter_u8(self, n0: int, n: int) -> Iterator[np.ndarray]:
        """The uint8 stream for absolute samples [n0, n0 + n), in order."""
        todo = iter(tasks(n0, n, self.fs, self.task_s))
        pending: deque[Future[np.ndarray]] = deque(
            self._ex.submit(_task, a, k) for a, k in itertools.islice(todo, 2 * self.workers)
        )
        try:
            while pending:
                u = pending.popleft().result()
                nxt = next(todo, None)
                if nxt is not None:
                    pending.append(self._ex.submit(_task, nxt[0], nxt[1]))
                yield u
        finally:
            for f in pending:
                f.cancel()

    def close(self) -> None:
        self._ex.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> RenderPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def iter_u8(
    canonical: str, base: Path, n0: int, n: int, fs: int, workers: int, task_s: int = TASK_S
) -> Iterator[np.ndarray]:
    """The uint8 stream for absolute samples [n0, n0 + n), rendered by up to `workers` processes, in order."""
    nw = min(workers, len(tasks(n0, n, fs, task_s)))
    if nw < 1:
        return
    with RenderPool(canonical, base, nw, fs, task_s) as pool:
        yield from pool.iter_u8(n0, n)
