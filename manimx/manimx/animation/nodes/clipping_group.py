"""ClippingGroup: content shown only inside a mask; it snapshots to an
RClippingGroup."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from ...robjects import RClippingGroup, RNode
from ..signals.kinds import DISCRETE
from ..signals.signal import RootSignal, Signal
from .node import (
    ABS,
    ChangeList,
    Node,
    PositionMixin,
    RotationMixin,
    ScalableMixin,
    animatable,
    find,
    find_all,
)


class ClippingGroup(PositionMixin, RotationMixin, ScalableMixin, Node):
    """Shows ``content`` only inside ``mask`` (the renderer's soft alpha
    mask: the mask's fill and stroke opacity, as drawn, multiply the
    content). The mask itself is never drawn. Transform and alpha apply to
    both, so scaling the group scales the mask with its content."""

    def __init__(
        self, mask: Node, content: Node, name: str | None = None, one_sided: bool = False, tags: Iterable[str] = ()
    ) -> None:
        super().__init__(name, tags=tags)
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

    def _build_snapshot(self) -> RNode | None:
        if self._hidden():
            return None
        mask, content = self.mask.snapshot(), self.content.snapshot()
        if mask is None or content is None:
            return None  # nothing to show through
        return RClippingGroup(mask=mask, content=content, one_sided=bool(self.one_sided.value), transform=self._transform(), alpha=self.alpha.value, uid=self.uid)


__all__ = ["ClippingGroup"]
