"""Group: a node that only holds others; it snapshots to an RGroup."""

from __future__ import annotations

from collections.abc import Iterator

from ..objects import RGroup, RNode
from .node import Node, PositionMixin, RotationMixin, ScalableMixin
from .signals import Signal


class Group(PositionMixin, RotationMixin, ScalableMixin, Node):
    """Holds other nodes; its transform and alpha apply to all of them."""

    def __init__(self, *children: Node, name: str | None = None) -> None:
        super().__init__(name)
        self.children = list(children)

    def add(self, *children: Node) -> Group:
        self.children.extend(children)
        return self

    def family(self) -> Iterator[Node]:
        yield self
        for child in self.children:
            yield from child.family()

    def snapshot(self, alphas: tuple[Signal, ...] = ()) -> RNode | None:
        if self._hidden():
            return None
        alphas = (*alphas, self.alpha)
        children = [snapshot for child in self.children if (snapshot := child.snapshot(alphas)) is not None]
        return RGroup(children=children, transform=self._transform(), uid=self.uid)


__all__ = ["Group"]
