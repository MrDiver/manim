from __future__ import annotations

from dataclasses import dataclass, field

from .render_object import RenderObject


@dataclass(slots=True, eq=False)
class Group(RenderObject):
    """Invisible container: no geometry or style, only children.

    The group's transform and visibility apply to the whole subtree.
    """

    children: list[RenderObject] = field(default_factory=list)
