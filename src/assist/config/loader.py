"""Scenario loading (plan §3.3): YAML/JSON -> includes -> `--set` -> band defaults -> presets.

Random expressions are drawn afterwards (`randexpr`) and the result is validated (`schema`).

- YAML is read as YAML 1.2 numbers (`137.0e6` is a float, as in JSON) and timestamps stay strings.
- `include:` paths are relative to the including file, then to the package (`bands/vhf137/...`).
- Merging: dicts merge recursively; lists of dicts with `id`s merge by id; other lists are replaced. A dict whose
  discriminator (`type`, `dist`, `kind`) changes is replaced, not merged, and giving one of a set of alternatives
  drops the others (a level's p_tx_dbw / cn0_zenith_dbhz / cal; an orbit's norad / kepler / synthetic_pass).
- `--set path=value` addresses list items by id: `sources[z1].freq_hz=137.35e6`. It runs before presets are
  expanded, so it can override any preset default through the source.
- Presets: a source's keys override the preset. Source keys that name an emitter (`gateway: {...}`) go to that
  emitter; emitter fields given on the source (`freq_hz`, `schedule`, ...) go to every emitter.
"""

import copy
import json
import re
from pathlib import Path
from typing import Any

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent.parent
DISCRIMINATORS = ("type", "dist", "kind")
EXCLUSIVE = ({"p_tx_dbw", "cn0_zenith_dbhz", "cal"}, {"norad", "kepler", "synthetic_pass"})
SOURCE_KEYS = {"id", "preset", "platform", "name", "orbit", "in_catalog", "catalog_norad", "emitters"}
EMITTER_KEYS = {
    "id",
    "enabled",
    "signal",
    "freq_hz",
    "offset_hz",
    "oscillator",
    "schedule",
    "am",
    "level",
    "pattern",
}


class ConfigError(ValueError):
    pass


class _Loader(yaml.SafeLoader):
    pass


# YAML 1.2 floats: PyYAML (YAML 1.1) reads `137.0e6` and `1e6` as strings.
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)
# Keep ISO timestamps as strings (times are parsed by the schema, always with an explicit zone).
_Loader.yaml_implicit_resolvers = {
    k: [(tag, rx) for tag, rx in v if tag != "tag:yaml.org,2002:timestamp"]
    for k, v in _Loader.yaml_implicit_resolvers.items()
}


def parse_yaml(text: str) -> Any:
    return yaml.load(text, Loader=_Loader)


def read_document(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text()
    d = json.loads(text) if p.suffix.lower() == ".json" else parse_yaml(text)
    if not isinstance(d, dict):
        raise ConfigError(f"{p}: expected a mapping at the top level")
    return d


def _has_ids(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(x, dict) and "id" in x for x in v)


def merge_by_id(base: list[dict[str, Any]], over: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [copy.deepcopy(x) for x in base]
    index = {x["id"]: i for i, x in enumerate(out)}
    for x in over:
        if x["id"] in index:
            out[index[x["id"]]] = deep_merge(out[index[x["id"]]], x)
        else:
            index[x["id"]] = len(out)
            out.append(copy.deepcopy(x))
    return out


def deep_merge(base: Any, over: Any) -> Any:
    """`over` on top of `base` (neither is modified)."""
    if isinstance(base, dict) and isinstance(over, dict):
        if any(k in base and k in over and base[k] != over[k] for k in DISCRIMINATORS):
            return copy.deepcopy(over)
        out = copy.deepcopy(base)
        for group in EXCLUSIVE:  # choosing one alternative drops the others (level: cal replaces cn0_zenith_dbhz)
            if any(over.get(k) is not None for k in group):
                for k in group - set(over):
                    out.pop(k, None)
        for k, v in over.items():
            out[k] = deep_merge(out[k], v) if k in out else copy.deepcopy(v)
        return out
    if _has_ids(base) and _has_ids(over) and base and over:
        return merge_by_id(base, over)
    return copy.deepcopy(over)


def find_include(name: str, base_dir: Path) -> Path:
    for cand in (base_dir / name, PACKAGE_DIR / name):
        if cand.is_file():
            return cand
    raise ConfigError(f"include {name!r} not found (looked in {base_dir} and the package)")


def load_with_includes(path: str | Path, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    p = Path(path).resolve()
    if p in _stack:
        raise ConfigError(f"include cycle: {' -> '.join(str(x) for x in (*_stack, p))}")
    doc = read_document(p)
    includes = doc.pop("include", None) or []
    if isinstance(includes, str):
        includes = [includes]
    merged: dict[str, Any] = {}
    for name in includes:
        merged = deep_merge(merged, load_with_includes(find_include(str(name), p.parent), (*_stack, p)))
    return deep_merge(merged, doc)


_SEGMENT = re.compile(r"([^.\[\]]+)((?:\[[^\]]+\])*)")


def parse_path(path: str) -> list[str | tuple[str]]:
    """`a.b[x][2].c` -> ['a', 'b', ('x',), ('2',), 'c']: plain keys, and list selectors as 1-tuples."""
    out: list[str | tuple[str]] = []
    for part in path.split("."):
        m = _SEGMENT.fullmatch(part)
        if not m:
            raise ConfigError(f"bad path {path!r}")
        out.append(m.group(1))
        out += [(s,) for s in re.findall(r"\[([^\]]+)\]", m.group(2))]
    return out


def _select(lst: list[Any], sel: str, path: str) -> int:
    for i, x in enumerate(lst):
        if isinstance(x, dict) and str(x.get("id")) == sel:
            return i
    if sel.isdigit() and int(sel) < len(lst):
        return int(sel)
    raise ConfigError(f"{path}: no list item with id {sel!r}")


def apply_set(doc: dict[str, Any], assignment: str) -> None:
    """Apply one `path=value` override in place; the value is parsed as YAML (`137.35e6`, `true`, `[1, 2]`)."""
    if "=" not in assignment:
        raise ConfigError(f"--set needs path=value, got {assignment!r}")
    path, raw = assignment.split("=", 1)
    keys = parse_path(path.strip())
    value = parse_yaml(raw)
    node: Any = doc
    for i, k in enumerate(keys):
        last = i == len(keys) - 1
        if isinstance(k, tuple):
            if not isinstance(node, list):
                raise ConfigError(f"{path}: [{k[0]}] applied to a non-list")
            j = _select(node, k[0], path)
            if last:
                node[j] = value
            else:
                node = node[j]
        else:
            if not isinstance(node, dict):
                raise ConfigError(f"{path}: {k!r} applied to a non-mapping")
            if last:
                node[k] = value
            else:
                nxt = keys[i + 1]
                if k not in node:
                    if isinstance(nxt, tuple):
                        raise ConfigError(f"{path}: {k!r} does not exist")
                    node[k] = {}
                node = node[k]


def band_dir(name: str) -> Path:
    d = PACKAGE_DIR / "bands" / name
    if not (d / "band.yaml").is_file():
        raise ConfigError(f"unknown band {name!r}")
    return d


def load_band(name: str) -> dict[str, Any]:
    return read_document(band_dir(name) / "band.yaml")


def load_preset(name: str) -> dict[str, Any]:
    p = PACKAGE_DIR / "presets" / f"{name}.yaml"
    if not p.is_file():
        known = sorted(x.stem for x in (PACKAGE_DIR / "presets").glob("*.yaml"))
        raise ConfigError(f"unknown preset {name!r} (known: {', '.join(known)})")
    return read_document(p)


def expand_source(src: dict[str, Any]) -> dict[str, Any]:
    """Apply a source's overrides to its preset (or to its own emitters when it has no preset)."""
    sid = src.get("id", "?")
    base = load_preset(src["preset"]) if src.get("preset") else {}
    out = {k: copy.deepcopy(v) for k, v in base.items() if k != "emitters"}
    emitters = [copy.deepcopy(e) for e in base.get("emitters", [])]
    if "emitters" in src:
        if not _has_ids(src["emitters"]):
            raise ConfigError(f"source {sid}: every emitter needs an id")
        emitters = merge_by_id(emitters, src["emitters"])
    ids = {e["id"] for e in emitters}
    for k, v in src.items():
        if k == "emitters":
            continue
        if k in SOURCE_KEYS:
            out[k] = deep_merge(out[k], v) if k in out else copy.deepcopy(v)
        elif k in ids:
            emitters = [deep_merge(e, v) if e["id"] == k else e for e in emitters]
        elif k in EMITTER_KEYS:
            emitters = [deep_merge(e, {k: v}) if k in e else {**e, k: copy.deepcopy(v)} for e in emitters]
        else:
            raise ConfigError(f"source {sid}: unknown key {k!r}")
    out["emitters"] = emitters
    return out


def load_scenario(path: str | Path, sets: tuple[str, ...] | list[str] = ()) -> dict[str, Any]:
    """The merged document before random expressions: includes, --set, band defaults, presets."""
    doc = load_with_includes(path)
    for s in sets:
        apply_set(doc, s)
    band = load_band(str(doc.get("band", "vhf137")))
    doc = deep_merge(band.get("defaults", {}), doc)
    doc["sources"] = [expand_source(s) for s in doc.get("sources", [])]
    return doc
