"""ClippingGroup: content shown only inside a mask; it snapshots to an
RClippingGroup."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from ..robjects import RClippingGroup, RNode
from .node import Node, PositionMixin, RotationMixin, ScalableMixin, find, find_all
from .signals import Signal


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
        self.one_sided = one_sided

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

    def snapshot(self, alphas: tuple[Signal, ...] = ()) -> RNode | None:
        if self._hidden():
            return None
        alphas = (*alphas, self.alpha)
        mask, content = self.mask.snapshot(alphas), self.content.snapshot(alphas)
        if mask is None or content is None:
            return None  # nothing to show through
        return RClippingGroup(mask=mask, content=content, one_sided=self.one_sided, transform=self._transform(), uid=self.uid)


__all__ = ["ClippingGroup"]
