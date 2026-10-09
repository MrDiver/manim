from __future__ import annotations

from dataclasses import dataclass, field

from .render_object import RNode


@dataclass(slots=True, eq=False)
class RGroup(RNode):
    """Invisible container: no geometry or style, only children.

    The group's transform and visibility apply to the whole subtree.
    """

    children: list[RNode] = field(default_factory=list)
