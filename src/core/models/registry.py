"""Model registry: name → class lookup + build_model dispatcher."""

from __future__ import annotations


class Registry:
    def __init__(self):
        self._d: dict[str, type] = {}

    def register(self, name: str, cls: type) -> None:
        self._d[name] = cls

    def get(self, name: str) -> type:
        if name not in self._d:
            raise KeyError(f"Unknown model {name!r}. Known: {sorted(self._d)}")
        return self._d[name]

    def has(self, name: str) -> bool:
        return name in self._d

    def names(self) -> list[str]:
        return sorted(self._d)


_GLOBAL = Registry()


def get_registry() -> Registry:
    return _GLOBAL


def build_model(cfg, name: str):
    """Instantiate a model by name. Tree → registry; DL/surrogate → cfg.dl_factory."""
    if _GLOBAL.has(name):
        return _GLOBAL.get(name)()
    if cfg.dl_factory is not None:
        m = cfg.dl_factory(name)
        if m is None:
            raise KeyError(f"cfg.dl_factory({name!r}) returned None")
        return m
    raise KeyError(f"Unknown model {name!r}. Tree: {_GLOBAL.names()}; "
                   f"DL: requires cfg.dl_factory")
