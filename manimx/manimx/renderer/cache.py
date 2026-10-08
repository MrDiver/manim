"""Per-object data cache shared by renderer backends.

See ``docs/render_engine_target_architecture.md`` section 9
("MobjectDataCache"). The cache is a dictionary addressed by IDs only, never by
live objects:

    (uid, kind) -> (version, data)

* ``uid`` is the runtime-unique :attr:`RenderObject.uid`.
* ``kind`` names what was derived, e.g. ``"fill-mesh"``. One object can own
  many kinds of data.
* ``version`` is whatever the producer derived the data from, usually
  ``obj.version("points")``. A lookup with a different version is a
  miss, and the next ``put`` replaces the stale entry, so each object keeps at
  most one entry per kind.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from .commands import CacheCommand

T = TypeVar("T")

_MISS = object()


class MobjectDataCache:
    def __init__(self) -> None:
        self._entries: dict[tuple[int, str], tuple[Hashable, Any]] = {}

    def get(self, uid: int, kind: str, version: Hashable) -> Any | None:
        """Cached data, or ``None`` if missing or derived from another version."""
        entry = self._entries.get((uid, kind))
        if entry is None or entry[0] != version:
            return None
        return entry[1]

    def put(self, uid: int, kind: str, version: Hashable, data: Any) -> None:
        self._entries[(uid, kind)] = (version, data)

    def get_or_create(
        self, uid: int, kind: str, version: Hashable, create: Callable[[], T]
    ) -> T:
        """Return cached data or build it with ``create()`` and cache it."""
        entry = self._entries.get((uid, kind), _MISS)
        if entry is not _MISS and entry[0] == version:  # type: ignore[index]
            return entry[1]  # type: ignore[index]
        data = create()
        self._entries[(uid, kind)] = (version, data)
        return data

    def execute(self, command: CacheCommand) -> None:
        """Apply a lifecycle command, see :mod:`.commands`."""
        command.apply(self)

    def invalidate(self, uid: int) -> None:
        """Drop every entry of one object, e.g. when it is deleted."""
        for key in [key for key in self._entries if key[0] == uid]:
            del self._entries[key]

    def retain(self, uids: Iterable[int]) -> None:
        """Drop entries of all objects not in ``uids`` (garbage collection)."""
        keep = set(uids)
        for key in [key for key in self._entries if key[0] not in keep]:
            del self._entries[key]

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
