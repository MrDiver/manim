"""Group: a node that only holds others; it snapshots to an RGroup."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

from ..robjects import RGroup, RNode
from .node import Node, PositionMixin, RotationMixin, ScalableMixin, find, find_all


class Group(PositionMixin, RotationMixin, ScalableMixin, Node):
    """Holds other nodes; its transform and alpha apply to all of them."""

    def __init__(self, *children: Node, name: str | None = None, tags: Iterable[str] = ()) -> None:
        super().__init__(name, tags=tags)
        self.children = list(children)
        self._structure_listeners: list[Callable[[Group], None]] = []

    def add(self, *children: Node) -> Group:
        self.children.extend(children)
        for listener in list(self._structure_listeners):
            listener(self)
        return self

    def on_structure_change(self, listener: Callable[[Group], None]) -> None:
        """Call ``listener(group)`` whenever children are added. Membership
        is not a signal (it does not change over time; presence does), so
        signals computed from a family, such as bounds(), listen here and
        rebind their inputs while the scene is composed."""
        self._structure_listeners.append(listener)

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

    def snapshot(self) -> RNode | None:
        """Like :meth:`Node.snapshot`, and the children's snapshots are
        part of the token: a group is reused while its own signals hold
        and every child returned the same render node as before."""
        if self._hidden():
            return None
        children = [snapshot for child in self.children if (snapshot := child.snapshot()) is not None]
        # The previous snapshot keeps its children alive, so equal ids mean the same objects.
        token = (self._snapshot_token(), tuple(map(id, children)))
        last = self._last_snapshot
        if last is not None and last[0] == token:
            return last[1]
        snapshot = RGroup(children=children, transform=self._transform(), alpha=self.alpha.value, uid=self.uid)
        self._last_snapshot = (token, snapshot)
        return snapshot


__all__ = ["Group"]
