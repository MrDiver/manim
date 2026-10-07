"""Turn a render-object tree into a flat, ordered list of drawable leaves.

This is the only place where the hierarchy matters to a renderer: groups
contribute their transform and visibility to their subtree and then
disappear. Backends only ever see :class:`DrawItem` s.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from ..objects import ClippingGroup, Group, RenderObject, Shape2D, Shape3D


@dataclass(frozen=True, slots=True, eq=False)
class DrawItem:
    """One drawable leaf with everything resolved from its ancestors."""

    obj: Shape2D | Shape3D
    world: np.ndarray
    """Local-to-world 4x4 matrix: product of all ancestor transforms."""
    order: int
    """Position in depth-first tree order (Manim's painter's order)."""


def flatten(root: RenderObject) -> list[DrawItem]:
    """Depth-first list of visible leaves in painter's order."""
    return [
        DrawItem(obj, world, order)
        for order, (obj, world) in enumerate(_walk(root, np.eye(4)))
    ]


def bounding_box(root: RenderObject) -> tuple[np.ndarray, np.ndarray]:
    """World-space ``(min, max)`` corners over all anchors of all 2D leaves.

    Like Manim, only anchors count (not handles). Useful to center a group,
    e.g. Manim's ``move_to(ORIGIN)``.
    """
    anchors = [
        (item.world @ np.column_stack([anchors, np.zeros(len(anchors)), np.ones(len(anchors))]).T).T[:, :3]
        for item in flatten(root)
        if isinstance(item.obj, Shape2D)
        and len(anchors := item.obj.curves[:, [0, 3]].reshape(-1, 2))
    ]
    if not anchors:
        return np.zeros(3), np.zeros(3)
    points = np.concatenate(anchors)
    return points.min(axis=0), points.max(axis=0)


def _walk(
    obj: RenderObject, parent_world: np.ndarray
) -> Iterator[tuple[Shape2D | Shape3D, np.ndarray]]:
    if not obj.visible:
        return
    # Child transforms compose with the parent: world = parent @ local.
    world = parent_world @ obj.transform.matrix()
    if isinstance(obj, (Shape2D, Shape3D)):
        yield obj, world
    elif isinstance(obj, Group):
        for child in obj.children:
            yield from _walk(child, world)
    elif isinstance(obj, ClippingGroup):
        raise NotImplementedError("ClippingGroup is not supported by renderers yet")
    else:
        raise TypeError(f"unsupported render object: {type(obj).__name__}")
