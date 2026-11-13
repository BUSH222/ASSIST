"""Registries of the extension points (PLAN.md): named classes, core and external.

A registry maps names to classes. Core classes (defined inside the `assist` package) are always available; their
modules are imported the first time the registry is asked for anything. External classes are *declared* when their
module is imported (the `register` decorator records them) but become available only while a scenario that lists
their plugin is active: `assist.plugins.activate` installs exactly the declared set and removes the rest. An installed
package is therefore never used by a scenario that does not name it.
"""

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

CORE = "core"


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class Declared:
    """One `register(name)` call of an external module."""

    category: str
    name: str
    cls: type
    module: str


# Every external registration, in the order the decorators ran (a module imported twice declares once).
_declared: list[Declared] = []


def declared_by(module: str) -> list[Declared]:
    """What the module `module` and its submodules registered."""
    return [d for d in _declared if d.module == module or d.module.startswith(module + ".")]


def is_core_class(cls: type) -> bool:
    return cls.__module__ == "assist" or cls.__module__.startswith("assist.")


class Registry[T: type]:
    def __init__(self, category: str, core_modules: tuple[str, ...]) -> None:
        self.category = category
        self._noun = " kind" if category == "signal" else ""
        self._core_modules = core_modules
        self._core: dict[str, T] = {}
        self._external: dict[str, tuple[T, str]] = {}  # name -> (class, origin)
        self._loaded = False

    def register(self, name: str) -> Callable[[T], T]:
        """Class decorator. Core classes are registered at once; external ones are declared (see the module text)."""
        if not name or not isinstance(name, str):
            raise RegistryError(f"a {self.category} name must be a non-empty string, got {name!r}")

        def deco(cls: T) -> T:
            if is_core_class(cls):
                other = self._core.get(name)
                if other is not None and other is not cls:
                    raise RegistryError(f"{self.category} {name!r} is registered twice ({other.__module__})")
                self._core[name] = cls
            elif not any(d.cls is cls and d.name == name and d.category == self.category for d in _declared):
                _declared.append(Declared(self.category, name, cls, cls.__module__))
            return cls

        return deco

    def _ensure_core(self) -> None:
        if not self._loaded:
            self._loaded = True
            for m in self._core_modules:
                importlib.import_module(m)
            # A fixed order (the module list's), whatever was imported first elsewhere.
            rank = {m: i for i, m in enumerate(self._core_modules)}
            order = sorted(self._core, key=lambda n: rank.get(self._core[n].__module__, len(rank)))
            self._core = {n: self._core[n] for n in order}

    def get(self, name: str) -> T:
        self._ensure_core()
        hit = self._core.get(name)
        if hit is not None:
            return hit
        ext = self._external.get(name)
        if ext is not None:
            return ext[0]
        raise RegistryError(
            f"unknown {self.category}{self._noun} {name!r} (known: {', '.join(self.names())}); "
            f"a {self.category} from an external module needs that module in the scenario's `plugins:` list"
        )

    def __contains__(self, name: object) -> bool:
        self._ensure_core()
        return name in self._core or name in self._external

    def names(self) -> list[str]:
        """Core names in registration order, then the active external ones."""
        self._ensure_core()
        return [*self._core, *self._external]

    def origin(self, name: str) -> str:
        """`core`, or the plugin that provides the name."""
        self.get(name)
        return CORE if name in self._core else self._external[name][1]

    def is_external(self, name: str) -> bool:
        return self.origin(name) != CORE

    def core_names(self) -> list[str]:
        self._ensure_core()
        return list(self._core)

    def set_external(self, items: Iterable[tuple[str, T, str]]) -> None:
        """Replace the active external classes by (name, class, origin) items; a clash is an error naming both."""
        self._ensure_core()
        new: dict[str, tuple[T, str]] = {}
        for name, cls, origin in items:
            if name in self._core:
                raise RegistryError(
                    f"plugin {origin}: {self.category} {name!r} is a core name; give the plugin entry a `prefix:`"
                )
            if name in new and new[name][0] is not cls:
                raise RegistryError(
                    f"plugins {new[name][1]} and {origin} both register {self.category} {name!r}; "
                    "give one of the entries a `prefix:` or an `only:` list"
                )
            new[name] = (cls, origin)
        self._external = new


signals: Registry[Any] = Registry(
    "signal",
    (
        "assist.signals.cw",
        "assist.signals.oqpsk",
        "assist.signals.oqpsk_burst",
        "assist.signals.sdpsk",
        "assist.signals.lora",
        "assist.signals.d8psk",
        "assist.signals.gfsk",
        "assist.signals.am",
        "assist.signals.noise",
    ),
)

platforms: Registry[Any] = Registry("platform", ("assist.platforms.satellite", "assist.platforms.fixed"))

sinks: Registry[Any] = Registry("sink", ("assist.sinks.wavfile", "assist.sinks.fft", "assist.sinks.rtltcp"))

ALL: dict[str, Registry[Any]] = {"signal": signals, "platform": platforms, "sink": sinks}
