"""Group: a node that only holds others; it snapshots to an RGroup."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

from .node import Node, PositionMixin, RotationMixin, ScalableMixin, adopt, find, find_all


class Group(PositionMixin, RotationMixin, ScalableMixin, Node):
    """Holds other nodes; its transform and alpha apply to all of them."""

    render_kind = "group"

    def __init__(self, *children: Node, name: str | None = None, tags: Iterable[str] = ()) -> None:
        super().__init__(name, tags=tags)
        self.children: list[Node] = []
        self._structure_listeners: list[Callable[[Group], None]] = []
        self.add(*children)

    def add(self, *children: Node) -> Group:
        """Put ``children`` at the end of the group; each must not be in a
        group yet (a node has one parent, see ``adopt``)."""
        for child in children:
            adopt(self, child)
            self.children.append(child)
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

    def render_children(self) -> list[Node]:
        return self.children


__all__ = ["Group"]
