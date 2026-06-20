"""A tiny name->factory registry so new arms are config-only.

Adding an intervention arm is: write a factory, decorate it with @ARMS.register("name"), reference
it by name in a config's `sweep.arms`. No edits to the orchestrator. Pure stdlib.
"""

from typing import Callable, Dict, List


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Callable] = {}

    def register(self, name: str) -> Callable:
        def deco(fn: Callable) -> Callable:
            if name in self._items:
                raise KeyError(f"{self.kind} {name!r} already registered")
            self._items[name] = fn
            return fn
        return deco

    def get(self, name: str) -> Callable:
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} {name!r}; known: {self.names()}")
        return self._items[name]

    def names(self) -> List[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


ARMS = Registry("arm")
