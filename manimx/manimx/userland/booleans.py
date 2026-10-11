"""Boolean shapes: the union, intersection, difference or exclusion of
the filled areas of other nodes.

``Union(circle, square)`` is a Shape2D whose points are derived from the
inputs' points and transforms (and those of their children: a Typst
formula or a Group counts as the area of all its paths). It follows the
inputs while they move or morph; the inputs need not be in the scene.
The outline is computed with Skia's path operations
(:func:`~manimx.userland.geometry.boolean_points`), only when an input
changes.

The inputs lie in the boolean shape's parent space as they would as its
siblings; their z is dropped. The members of each input are taken as
they are when the shape is made: a child added to an input group later
does not count.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..animation import Node, Shape2D
from .geometry import boolean_points
from .markers import _family_layout, family_paths


class _Boolean(Shape2D):
    """A Shape2D whose outline is ``operation`` applied to ``nodes``."""

    operation = "union"

    def __init__(self, *nodes: Node, name: str | None = None, **style: Any) -> None:
        if len(nodes) < 2:
            raise ValueError(f"{type(self).__name__} needs at least two nodes")
        super().__init__((), name, **style)
        self.nodes = nodes
        layouts, inputs = [], []
        for node in nodes:
            layout, signals = _family_layout(node)
            layouts.append((layout, len(signals)))
            inputs.extend(signals)
        self._derive_points(_BooleanPoints(self.operation, tuple(layouts)), *inputs)


class Union(_Boolean):
    """The area covered by any of ``nodes``."""

    operation = "union"


class Intersection(_Boolean):
    """The area covered by all of ``nodes``."""

    operation = "intersection"


class Difference(_Boolean):
    """The area of the first node not covered by any of the others."""

    operation = "difference"


class Exclusion(_Boolean):
    """The area covered by an odd number of ``nodes`` (two nodes: by
    exactly one of them)."""

    operation = "exclusion"


class _BooleanPoints:
    """The derivation of a boolean shape's points: per input node, its
    family layout and how many of the values are its inputs."""

    __slots__ = ("layouts", "operation")

    def __init__(self, operation: str, layouts: tuple[tuple[Any, int], ...]) -> None:
        self.operation, self.layouts = operation, layouts

    def __call__(self, *values: Any) -> np.ndarray:
        operands: list[Sequence[np.ndarray]] = []
        cursor = 0
        for layout, count in self.layouts:
            operands.append([points[:, :2] for points in family_paths(layout, values[cursor : cursor + count])])
            cursor += count
        return boolean_points(self.operation, operands)


__all__ = ["Difference", "Exclusion", "Intersection", "Union"]
