"""Index labels: which number each path of a group has, on the frame.

For MorphMatching's pairs (and finding a glyph in a formula), the page
can show, over the frame, the index of every Shape2D and Curve3D among
the paths of the group that holds it, and its name. Like Manim's
``index_labels``, but drawn by the page over the image instead of being
added to the scene: the scene stays as it is, and the labels stay sharp
at any size.

A label is placed at the center of its path's bounding box, projected
through the clip's camera, as fractions of the frame (x to the right,
y down), and carries the path's color.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..animation import Curve3D, Group, Node, Shape2D
from ..animation.timing import Timeline
from ..robjects import RBezierPath, RNode


def index_labels(timeline: Timeline, time: float, aspect_ratio: float) -> list[dict[str, Any]]:
    """The labels of every path drawn at ``time``, in layer order."""
    views = timeline.frame_at(time)  # resolves the clips' scenes first
    groups = {node.uid: node for clip in timeline.clips for root in clip.scene.roots for node in root.family() if isinstance(node, Group)}
    labels: list[dict[str, Any]] = []
    for view in views:
        clip_space = view.camera.projection_matrix(aspect_ratio) @ view.camera.view_matrix()
        _visit(view.root, np.eye(4), None, groups, clip_space, labels)
    return labels


def _visit(node: RNode, parent: np.ndarray, group: Group | None, groups: dict[int, Group], clip_space: np.ndarray, labels: list[dict[str, Any]]) -> None:
    world = parent @ node.transform.matrix()
    if isinstance(node, RBezierPath) and group is not None and len(node.points):
        paths = [child for child in group.children if isinstance(child, (Shape2D, Curve3D))]
        index = next((i for i, path in enumerate(paths) if path.uid == node.uid), None)
        if index is not None:
            labels.append(_label(node, world, clip_space, index, paths[index], group))
    # Children are labeled within the group that holds them directly.
    for child in getattr(node, "children", ()):
        _visit(child, world, groups.get(node.uid), groups, clip_space, labels)


def _label(node: RBezierPath, world: np.ndarray, clip_space: np.ndarray, index: int, path: Node, group: Group) -> dict[str, Any]:
    points = np.asarray(node.points, dtype=np.float64)
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    local = np.zeros(4)
    local[: len(center)], local[3] = center, 1.0
    clip = clip_space @ (world @ local)
    x, y = clip[:2] / clip[3]
    return {"x": (x + 1) / 2, "y": (1 - y) / 2, "index": index, "name": path.name, "group": group.name, "color": _color(node)}


def _color(node: RBezierPath) -> str:
    """The color the path shows most: its fill's where the fill is seen,
    else its stroke's (the first of a gradient), as ``#rrggbb``."""
    fill, stroke = getattr(node, "fill", None), getattr(node, "stroke", None)
    style = fill if fill is not None and np.max(fill.opacity) > 0 else stroke
    if style is None:
        return "#ffffff"
    red, green, blue = (round(255 * float(np.clip(channel, 0.0, 1.0))) for channel in np.asarray(style.color)[0][:3])
    return f"#{red:02x}{green:02x}{blue:02x}"


__all__ = ["index_labels"]
