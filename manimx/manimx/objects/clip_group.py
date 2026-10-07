from __future__ import annotations

from dataclasses import dataclass

from .render_object import RenderObject


@dataclass(slots=True, eq=False)
class ClippingGroup(RenderObject):
    """Invisible container that shows ``content`` only inside ``mask``.

    The mask contributes geometry only; it is never drawn itself.
    """

    mask: RenderObject | None = None
    content: RenderObject | None = None

    @property
    def children(self) -> list[RenderObject]:
        return [child for child in (self.mask, self.content) if child is not None]
