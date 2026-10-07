from __future__ import annotations

from dataclasses import dataclass

from .render_object import RenderObject


@dataclass(slots=True, eq=False)
class ClippingGroup(RenderObject):
    """Invisible container that shows ``content`` only inside ``mask``.

    The mask is never drawn itself; its alpha (fill and stroke as they would
    be drawn) multiplies the content's alpha.

    With ``one_sided=True`` a mask shape only counts where its front side
    (its local +z axis) faces the camera, which turns flat masks into
    portals: the faces of a cube seen from outside never overlap on screen,
    so each face can show its own world.

    Lights inside ``content`` only light objects inside this group (see
    :class:`Light`), so every clipped world can have its own lighting.
    """

    mask: RenderObject | None = None
    content: RenderObject | None = None
    one_sided: bool = False

    @property
    def children(self) -> list[RenderObject]:
        return [child for child in (self.mask, self.content) if child is not None]
