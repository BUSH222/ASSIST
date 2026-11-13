"""External modules: a scenario lists the plugins it uses under `plugins:`, and only those are loaded.

    plugins:
      - psk8                              # an entry point of the group `assist.plugins`, else an importable module
      - path: examples/plugins/pretty_gfsk.py   # a file: looked up in the working directory, then next to the scenario
      - module: mypkg.modems              # an importable module
        only: [psk8]                      # use only these of the names it registers
        prefix: lab                       # register them as lab.<name> (to get round a name clash)

A plugin is ordinary Python that imports `assist.api` and registers classes with its decorators. Nothing is scanned:
an installed package that no scenario names is never imported. `activate` makes exactly the listed plugins' classes
available and removes those of other scenarios, so a scenario means the same thing whatever else is installed.

Plugins run in this process with full trust, like any Python module a user chooses to import.

A plugin says which `assist.api` it was written for: `ASSIST_API = "1.0"` at the top level of its module. It loads
when the major version equals this package's `API_VERSION` and its minor is not newer (see `check_api`); a plugin
without the line is refused.

A file plugin is identified by its content: `activate` returns each entry with the file's `sha256`, the `api` the
plugin declares and a `version` (when the module or its distribution has one). The resolved scenario keeps them, so
the scenario hash changes when the plugin does, and a resolved scenario refuses a file that has changed since. A
scenario that lists no plugins has no `plugins` key in its canonical form: its hash does not depend on any of this.
"""

import hashlib
import importlib
import importlib.metadata
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from assist import registry

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from types import ModuleType

# Version of the plugin interface (`assist.api`), "major.minor". The minor grows when something is added (plugins
# written for an older minor keep working); the major grows when a plugin may have to change.
API_VERSION = "1.0"
API_ATTR = "ASSIST_API"  # the module attribute a plugin declares its version with

ENTRY_POINT_GROUP = "assist.plugins"
KEYS = {"name", "path", "module", "prefix", "only", "sha256", "version", "api"}


class PluginError(ValueError):
    pass


@dataclass(frozen=True)
class Loaded:
    label: str  # how messages name the plugin: its path, module or name as written
    module: ModuleType
    sha256: str | None  # file plugins: of the file
    version: str | None
    api: str = API_VERSION  # the assist.api version the plugin declares
    distribution: str | None = None  # the installed distribution it comes from, when there is one
    registered: tuple[tuple[str, str], ...] = ()  # (category, name as registered) of what this scenario uses of it


_files: dict[tuple[str, str], ModuleType] = {}  # (absolute path, sha256) -> module of a file plugin
_located: dict[tuple[str, str], str] = {}  # (path as written, sha256) -> absolute path it was found at
_active: list[Loaded] = []


def normalise(entry: Any) -> dict[str, Any]:
    """One `plugins:` entry as a dict with exactly one of `name`, `path`, `module` (a bare string is a name)."""
    if isinstance(entry, str):
        entry = {"name": entry}
    if not isinstance(entry, dict):
        raise PluginError(f"a plugins entry is a name or a mapping, got {entry!r}")
    e = {k: v for k, v in entry.items() if v is not None}
    unknown = sorted(set(e) - KEYS)
    if unknown:
        raise PluginError(f"plugins entry {entry!r}: unknown key(s) {', '.join(map(str, unknown))}")
    which = [k for k in ("name", "path", "module") if k in e]
    if len(which) != 1:
        raise PluginError(f"plugins entry {entry!r}: give exactly one of name, path, module")
    for k in ("name", "path", "module", "prefix", "sha256", "version", "api"):
        if k in e and (not isinstance(e[k], str) or not e[k]):
            raise PluginError(f"plugins entry {entry!r}: {k} must be a non-empty string")
    only = e.get("only")
    if only is not None and (not isinstance(only, list) or not only or not all(isinstance(x, str) for x in only)):
        raise PluginError(f"plugins entry {entry!r}: only must be a non-empty list of names")
    return e


def label_of(e: dict[str, Any]) -> str:
    return str(e.get("path") or e.get("module") or e.get("name"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_file(e: dict[str, Any], dirs: Sequence[Path]) -> Path:
    """The file of a `path:` entry: an absolute path as it is; a relative one in each of `dirs` in turn (the working
    directory, then the scenario's directory), after any place the same file was found before in this run."""
    raw = Path(e["path"]).expanduser()
    tried: list[Path] = []
    known = _located.get((e["path"], e.get("sha256", "")))
    cands: list[Path] = [Path(known)] if known else []
    cands += [raw] if raw.is_absolute() else [d / raw for d in dirs]
    for cand in cands:
        if cand.is_file():
            return cand.resolve()
        tried.append(cand)
    where = ", ".join(str(t) for t in dict.fromkeys(tried)) or "no directory to look in"
    raise PluginError(f"plugin {e['path']}: file not found (tried {where})")


def _load_file(e: dict[str, Any], dirs: Sequence[Path]) -> tuple[ModuleType, str]:
    path = find_file(e, dirs)
    sha = _sha256(path)
    want = e.get("sha256")
    if want is not None and want != sha:
        raise PluginError(
            f"plugin {e['path']}: {path} has changed since the scenario was resolved "
            f"(sha256 {sha[:12]}..., the scenario has {want[:12]}...); resolve the scenario again"
        )
    _located[(e["path"], sha)] = str(path)
    mod = _files.get((str(path), sha))
    if mod is None:
        name = f"assist_plugin_{''.join(c if c.isalnum() else '_' for c in path.stem)}_{sha[:12]}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise PluginError(f"plugin {e['path']}: {path} is not a Python module")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception as ex:
            del sys.modules[name]
            raise PluginError(f"plugin {e['path']}: {path} failed to import: {type(ex).__name__}: {ex}") from ex
        _files[(str(path), sha)] = mod
    return mod, sha


def _import(e: dict[str, Any], module: str) -> ModuleType:
    try:
        return importlib.import_module(module)
    except Exception as ex:
        raise PluginError(f"plugin {label_of(e)}: module {module} failed to import: {type(ex).__name__}: {ex}") from ex


def _entry_points(name: str) -> list[importlib.metadata.EntryPoint]:
    return list(importlib.metadata.entry_points(group=ENTRY_POINT_GROUP, name=name))


def _load_named(e: dict[str, Any]) -> tuple[ModuleType, tuple[str, str] | None]:
    """A `name:` entry: the entry point of that name in the group `assist.plugins`, else a module of that name.
    Entry points are only a name -> module table here; none is loaded unless a scenario names it.
    Returns the module and, for an entry point, its distribution's (name, version)."""
    eps = _entry_points(e["name"])
    if len(eps) > 1:
        dists = ", ".join(sorted(ep.dist.name if ep.dist else "?" for ep in eps))
        raise PluginError(f"plugin {e['name']}: several installed packages provide this entry point ({dists})")
    if eps:
        dist = eps[0].dist
        return _import(e, eps[0].module), (dist.name, dist.version) if dist else None
    if importlib.util.find_spec(e["name"].split(".")[0]) is None:
        raise PluginError(
            f"plugin {e['name']}: no `{ENTRY_POINT_GROUP}` entry point and no importable module of that name"
        )
    return _import(e, e["name"]), None


def _provenance(mod: ModuleType, dist: tuple[str, str] | None, is_file: bool) -> tuple[str | None, str | None]:
    """(version, distribution name) of a plugin: the module's `__version__` if it has one, else its distribution's."""
    if dist is None and not is_file:
        top = mod.__name__.split(".")[0]
        for name in importlib.metadata.packages_distributions().get(top, []):
            dist = (name, importlib.metadata.version(name))
            break
    v = getattr(mod, "__version__", None)
    version = v if isinstance(v, str) and v else (dist[1] if dist else None)
    return version, dist[0] if dist else None


def _parse_api(text: object) -> tuple[int, int] | None:
    if not isinstance(text, str):
        return None
    parts = text.split(".")
    if len(parts) != 2 or not all(p.isascii() and p.isdigit() for p in parts):
        return None
    return int(parts[0]), int(parts[1])


def check_api(label: str, declared: object) -> str:
    """The `ASSIST_API` of a plugin, checked against `API_VERSION`: the same major, and a minor not above ours."""
    if declared is None:
        raise PluginError(
            f"plugin {label}: the module does not say which assist.api it was written for; "
            f'add `{API_ATTR} = "{API_VERSION}"` to it'
        )
    want = _parse_api(declared)
    if want is None:
        raise PluginError(f'plugin {label}: {API_ATTR} must be a string "major.minor", got {declared!r}')
    have = _parse_api(API_VERSION)
    assert have is not None
    if want[0] != have[0]:
        raise PluginError(
            f"plugin {label} was written for assist.api {declared}; this is assist.api {API_VERSION} "
            "(another major version: the plugin has to be adapted)"
        )
    if want[1] > have[1]:
        raise PluginError(
            f"plugin {label} was written for assist.api {declared}; this is the older assist.api {API_VERSION} "
            "(update assist)"
        )
    assert isinstance(declared, str)
    return declared


def activate(entries: Iterable[Any] | None, dirs: Sequence[Path | None] = ()) -> list[dict[str, Any]]:
    """Load the listed plugins and make exactly their classes the active external ones (none listed: none active).

    `dirs`: where relative `path:` entries are looked up, in order. Returns the entries normalised, each with the
    `api` its plugin declares, file plugins with their `sha256`, and with a `version` where one is known."""
    if entries is not None and not isinstance(entries, list | tuple):
        raise PluginError(f"plugins must be a list, got {entries!r}")
    search = [d for d in dirs if d is not None]
    out: list[dict[str, Any]] = []
    loaded: list[Loaded] = []
    items: dict[str, list[tuple[str, type, str]]] = {cat: [] for cat in registry.ALL}
    seen: set[tuple[str, str | None, tuple[str, ...] | None]] = set()
    for raw in entries or ():
        e = normalise(raw)
        sha: str | None = None
        dist: tuple[str, str] | None = None
        if "path" in e:
            mod, sha = _load_file(e, search)
        elif "module" in e:
            mod = _import(e, e["module"])
        else:
            mod, dist = _load_named(e)
        label = label_of(e)
        ident = (mod.__name__, e.get("prefix"), tuple(e["only"]) if "only" in e else None)
        if ident in seen:
            raise PluginError(f"plugin {label} is listed twice")
        seen.add(ident)
        decl = registry.declared_by(mod.__name__)
        if not decl:
            raise PluginError(f"plugin {label}: the module registers nothing (no register_* decorator ran)")
        api = check_api(label, getattr(mod, API_ATTR, None))
        only = e.get("only")
        if only is not None:
            missing = sorted(set(only) - {d.name for d in decl})
            if missing:
                have = ", ".join(sorted({d.name for d in decl}))
                raise PluginError(f"plugin {label}: only: {', '.join(missing)} not registered by it (it has: {have})")
            decl = [d for d in decl if d.name in only]
        prefix = e.get("prefix")
        names = [(d.category, f"{prefix}.{d.name}" if prefix else d.name) for d in decl]
        for d, (cat, name) in zip(decl, names, strict=True):
            items[cat].append((name, d.cls, label))
        version, distribution = _provenance(mod, dist, sha is not None)
        if sha is not None:
            e["sha256"] = sha
        if version is not None:
            e["version"] = version
        e["api"] = api
        out.append(e)
        loaded.append(Loaded(label, mod, sha, version, api, distribution, tuple(names)))
    try:
        for cat, reg in registry.ALL.items():
            reg.set_external(items[cat])
    except registry.RegistryError as ex:
        for reg in registry.ALL.values():
            reg.set_external(())
        _active.clear()
        raise PluginError(str(ex)) from None
    _active[:] = loaded
    return out


def active() -> list[Loaded]:
    """The plugins of the scenario activated last."""
    return list(_active)


def located() -> dict[tuple[str, str], str]:
    """Where this process found its file plugins, to hand to worker processes (`remember`)."""
    return dict(_located)


def remember(found: dict[tuple[str, str], str]) -> None:
    _located.update(found)


def from_cli(value: str) -> dict[str, Any]:
    """A `--plugin` value: a file when it ends in .py or has a path separator, else a name."""
    return {"path": value} if value.endswith(".py") or "/" in value or "\\" in value else {"name": value}
