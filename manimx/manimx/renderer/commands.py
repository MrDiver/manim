"""Lifecycle commands sent to a renderer and its caches.

The renderer itself is stateless with respect to the scene: it renders
whatever tree it is given. Its caches, however, remember data per object
uid. When objects leave the scene, whoever manages the scene (later: the
manager in front of the renderer) sends a command:

    renderer.submit(RemoveObjects.subtree(group))

The renderer forwards every command to each cache it owns, so callers never
need to know which caches exist. New lifecycle events become new command
classes without touching the renderer or the caches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..objects import RClippingGroup, RGroup, RNode

if TYPE_CHECKING:
    from .cache import MobjectDataCache


class CacheCommand(ABC):
    """An operation applied to a :class:`MobjectDataCache`."""

    @abstractmethod
    def apply(self, cache: MobjectDataCache) -> None: ...


@dataclass(frozen=True, slots=True)
class RemoveObjects(CacheCommand):
    """Objects left the scene: drop everything cached for them."""

    uids: tuple[int, ...]

    @classmethod
    def subtree(cls, root: RNode) -> RemoveObjects:
        """Command removing ``root`` and all of its descendants."""
        return cls(tuple(obj.uid for obj in _descendants(root)))

    def apply(self, cache: MobjectDataCache) -> None:
        for uid in self.uids:
            cache.invalidate(uid)


@dataclass(frozen=True, slots=True)
class ClearScene(CacheCommand):
    """The whole scene was replaced: drop all cached data."""

    def apply(self, cache: MobjectDataCache) -> None:
        cache.clear()


def _descendants(obj: RNode) -> Iterator[RNode]:
    yield obj
    if isinstance(obj, (RGroup, RClippingGroup)):
        for child in obj.children:
            yield from _descendants(child)
