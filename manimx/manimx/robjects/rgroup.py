"""RGroup: the snapshot of a ``Group``."""

from __future__ import annotations

from dataclasses import dataclass, field

from .rnode import RNode


@dataclass(slots=True, eq=False)
class RGroup(RNode):
    """Invisible container: no geometry or style, only children.

    The group's transform and alpha apply to the whole subtree.
    """

    children: list[RNode] = field(default_factory=list)
