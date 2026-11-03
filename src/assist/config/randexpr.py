"""Random expressions (plan §3.3): `uniform(a,b)`, `normal(mu,sigma)`, `randint(a,b)` (inclusive),
`choice([..])`, `sample([..],k)`, `bernoulli(p)`.

Drawn at resolve from the stream ("resolve", source_id, field_path), where field_path is the path inside the
source with list items named by id (`emitters[carrier].freq_hz`). Fields outside sources use source_id "".
"""

import re
from typing import TYPE_CHECKING, Any

from assist.config.loader import ConfigError, parse_yaml

if TYPE_CHECKING:
    import numpy as np

    from assist.rng import KeyedRng

_EXPR = re.compile(r"^\s*(uniform|normal|randint|choice|sample|bernoulli)\s*\((.*)\)\s*$", re.S)


def is_expr(v: Any) -> bool:
    return isinstance(v, str) and _EXPR.match(v) is not None


def parse(s: str) -> tuple[str, list[Any]]:
    m = _EXPR.match(s)
    if not m:
        raise ConfigError(f"not a random expression: {s!r}")
    try:
        args = parse_yaml(f"[{m.group(2)}]")
    except Exception as e:
        raise ConfigError(f"bad arguments in {s!r}: {e}") from e
    return m.group(1), list(args)


def _num(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise ConfigError(f"{what}: expected a number, got {v!r}")
    return float(v)


def draw(s: str, g: np.random.Generator) -> Any:
    name, a = parse(s)
    want = {"uniform": 2, "normal": 2, "randint": 2, "choice": 1, "sample": 2, "bernoulli": 1}[name]
    if len(a) != want:
        raise ConfigError(f"{s!r}: {name} takes {want} argument(s)")
    if name == "uniform":
        lo, hi = _num(a[0], s), _num(a[1], s)
        if hi < lo:
            raise ConfigError(f"{s!r}: upper bound below lower bound")
        return float(g.uniform(lo, hi))
    if name == "normal":
        mu, sigma = _num(a[0], s), _num(a[1], s)
        if sigma < 0:
            raise ConfigError(f"{s!r}: sigma < 0")
        return float(g.normal(mu, sigma))
    if name == "randint":
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in a) or a[1] < a[0]:
            raise ConfigError(f"{s!r}: randint needs integers a <= b")
        return int(g.integers(a[0], a[1], endpoint=True))
    if name == "bernoulli":
        p = _num(a[0], s)
        if not 0 <= p <= 1:
            raise ConfigError(f"{s!r}: p must be in [0, 1]")
        return bool(g.random() < p)
    items = a[0]
    if not isinstance(items, list) or not items:
        raise ConfigError(f"{s!r}: needs a non-empty list")
    if name == "choice":
        return items[int(g.integers(len(items)))]
    k = a[1]
    if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= len(items):
        raise ConfigError(f"{s!r}: k must be an integer in 0..{len(items)}")
    return [items[int(i)] for i in g.choice(len(items), size=k, replace=False)]


def _walk(node: Any, path: str, source_id: str, rng: KeyedRng) -> Any:
    if isinstance(node, dict):
        return {k: _walk(v, f"{path}.{k}" if path else str(k), source_id, rng) for k, v in node.items()}
    if isinstance(node, list):
        out = []
        for i, v in enumerate(node):
            sel = v["id"] if isinstance(v, dict) and "id" in v else i
            out.append(_walk(v, f"{path}[{sel}]", source_id, rng))
        return out
    if is_expr(node):
        return draw(node, rng.stream("resolve", source_id, path))
    return node


def draw_all(doc: dict[str, Any], rng: KeyedRng) -> dict[str, Any]:
    """A copy of the document with every random expression drawn."""
    out = {k: _walk(v, k, "", rng) for k, v in doc.items() if k != "sources"}
    sources = []
    for src in doc.get("sources", []):
        sid = str(src.get("id", ""))
        sources.append({k: _walk(v, k, sid, rng) for k, v in src.items()})
    out["sources"] = sources
    return out
