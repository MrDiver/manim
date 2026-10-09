"""Group: a node that only holds others; it snapshots to an RGroup."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from ..objects import RGroup, RNode
from .node import Node, PositionMixin, RotationMixin, ScalableMixin, find, find_all
from .signals import Signal


class Group(PositionMixin, RotationMixin, ScalableMixin, Node):
    """Holds other nodes; its transform and alpha apply to all of them."""

    def __init__(self, *children: Node, name: str | None = None, tags: Iterable[str] = ()) -> None:
        super().__init__(name, tags=tags)
        self.children = list(children)

    def add(self, *children: Node) -> Group:
        self.children.extend(children)
        return self

    def find(self, name: str) -> Node:
        """The one node named ``name`` in this group's subtree; raises a
        LookupError (with close names) for none or several."""
        return find(self, name)

    def find_all(self, name: str | None = None, tags: Iterable[str] = ()) -> list[Node]:
        """Every node in this group's subtree named ``name`` (if given)
        that has all of ``tags``, in tree order; possibly empty."""
        return find_all(self, name, tags)

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
