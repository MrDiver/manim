"""ClippingGroup: content shown only inside a mask; it snapshots to an
RClippingGroup."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from ..signals.kinds import DISCRETE
from ..signals.signal import RootSignal, Signal
from .node import (
    ABS,
    ChangeList,
    Node,
    PositionMixin,
    RotationMixin,
    ScalableMixin,
    adopt,
    animatable,
    find,
    find_all,
)


class ClippingGroup(PositionMixin, RotationMixin, ScalableMixin, Node):
    """Shows ``content`` only inside ``mask`` (the renderer's soft alpha
    mask: the mask's fill and stroke opacity, as drawn, multiply the
    content). The mask itself is never drawn. Transform and alpha apply to
    both, so scaling the group scales the mask with its content."""

    render_kind = "clipping_group"

    def __init__(
        self, mask: Node, content: Node, name: str | None = None, one_sided: bool = False, tags: Iterable[str] = ()
    ) -> None:
        super().__init__(name, tags=tags)
        if mask is content:
            raise ValueError(f"{mask!r} cannot be both the mask and the content")
        adopt(self, mask)
        adopt(self, content)
        self.mask = mask
        self.content = content
        self.one_sided = RootSignal(one_sided, DISCRETE, name=f"{self.name}.one_sided")
        """``RClippingGroup.one_sided``: the mask only counts where its front
        faces the camera (portals)."""

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield self.one_sided

    @animatable
    def set_one_sided(self, one_sided: bool) -> ChangeList:
        return [(self.one_sided, ABS, one_sided)]

    def find(self, name: str) -> Node:
        """The one node named ``name`` in the mask or content (see Group.find)."""
        return find(self, name)

    def find_all(self, name: str | None = None, tags: Iterable[str] = ()) -> list[Node]:
        """Every matching node in the mask or content (see Group.find_all)."""
        return find_all(self, name, tags)

    def family(self) -> Iterator[Node]:
        yield self
        yield from self.mask.family()
        yield from self.content.family()

    def render_fields(self) -> Iterator[tuple[str, Signal]]:
        yield from super().render_fields()
        yield "one_sided", self.one_sided

    def render_children(self) -> list[Node]:
        """The mask, then the content."""
        return [self.mask, self.content]


__all__ = ["ClippingGroup"]
