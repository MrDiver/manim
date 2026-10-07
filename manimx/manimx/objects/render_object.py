from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from .transform import Quaternion, Transform
from .versioning import versioned_fields

__all__ = ["Quaternion", "RenderObject"]

# Runtime ids are never reused within a process, so renderer caches can key
# on them without ever confusing a new object with a deleted one.
_next_uid = itertools.count(1)


@dataclass(slots=True, eq=False)
class RenderObject:
    """Scene-graph node: identity, local 3D transform and visibility.

    Nodes compare by identity. Geometry and style live only on leaf objects
    such as :class:`Shape2D`; groups only arrange their children.

    :attr:`uid` is unique for the lifetime of the process. Fields declared
    with :func:`~.versioning.versioned` additionally count their assignments,
    see :meth:`version`.
    """

    id: str
    transform: Transform = field(default_factory=Transform)
    visible: bool = True
    uid: int = field(default_factory=lambda: next(_next_uid), init=False)
    _versions: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __setattr__(self, name: str, value: object) -> None:
        object.__setattr__(self, name, value)
        if name in versioned_fields(type(self)):
            try:
                versions = self._versions
            except AttributeError:  # assigned by __init__ before _versions exists
                return
            versions[name] = versions.get(name, 0) + 1

    def version(self, *fields: str) -> tuple[int, ...]:
        """Assignment count of each named versioned field (a cache key part)."""
        known = versioned_fields(type(self))
        for name in fields:
            if name not in known:
                raise ValueError(f"{type(self).__name__}.{name} is not versioned")
        return tuple(self._versions.get(name, 0) for name in fields)
