"""Turn a render-object tree into flat, ordered lists of drawable leaves.

This is the only place where the hierarchy matters to a renderer: groups
contribute their transform and visibility to their subtree and then
disappear. Backends only ever see :class:`DrawItem` s and
:class:`ClipScope` s.

Clipping: every visible :class:`RClippingGroup` with a mask opens a
:class:`ClipScope`. The leaves of its mask subtree are collected in the
scope (they form the mask and are not drawn themselves); the leaves of its
content subtree carry the index of the scope. Scopes nest: a mask inside
another scope's content is itself clipped by that outer scope, so the
innermost scope already describes the combined mask.

Lights are collected with their world matrices and the clip scope they
sit in, which limits them to objects of that scope (inside a mask they are
ignored, since masks only contribute alpha).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar

import numpy as np

from ..robjects import RClippingGroup, RGroup, RLight, RNode, RShape2D, RShape3D
from ..robjects.rbezier_path import RBezierPath

NO_CLIP = -1


@dataclass(frozen=True, slots=True, eq=False)
class DrawItem:
    """One drawable leaf with everything resolved from its ancestors."""

    obj: RBezierPath | RShape3D
    world: np.ndarray
    """Local-to-world 4x4 matrix: product of all ancestor transforms."""
    order: int
    """Position in depth-first tree order (Manim's painter's order)."""
    clip: int = NO_CLIP
    """Index of the innermost enclosing :class:`ClipScope`, or ``NO_CLIP``."""
    alpha: float = 1.0
    """Product of the ``alpha`` of the leaf and all its ancestors (>= 0)."""


@dataclass(slots=True, eq=False)
class ClipScope:
    """The mask of one :class:`RClippingGroup`."""

    group: RClippingGroup
    parent: int
    """Index of the enclosing scope (which also clips this mask), or ``NO_CLIP``."""
    mask: list[DrawItem] = field(default_factory=list)
    """Leaves forming the mask, drawn into the mask instead of the frame."""


@dataclass(frozen=True, slots=True, eq=False)
class LightItem:
    light: RLight
    world: np.ndarray
    clip: int = NO_CLIP
    """The clip scope the light is limited to, or NO_CLIP for everything."""
    alpha: float = 1.0
    """Product of the ``alpha`` of the light and its ancestors (>= 0);
    multiplies its intensity."""

T = TypeVar("T")


@dataclass(slots=True, eq=False)
class FlatScene:
    items: list[DrawItem] = field(default_factory=list)
    """Visible leaves in painter's order."""
    clips: list[ClipScope] = field(default_factory=list)
    """Clip scopes, every parent before its children."""
    lights: list[LightItem] = field(default_factory=list)

    def in_scope(self, clip: int, scope: int) -> bool:
        """Whether something in clip scope ``clip`` belongs to ``scope``
        (the scope itself or one nested in it). Everything is in NO_CLIP."""
        while True:
            if clip == scope or scope == NO_CLIP:
                return True
            if clip == NO_CLIP:
                return False
            clip = self.clips[clip].parent

    def shadow_lists(self, casters: list[tuple[T, int]]) -> tuple[dict[int, list[T]], dict[int, list[tuple[int, int]]]]:
        """The shadow casters ``(draw, clip)`` of each light scope (NO_CLIP
        first, then every clip scope): the draws, sorted by clip scope
        (unclipped first), and the runs ``(clip, count)`` of equal clip
        scopes among them. A shadow pass draws a clipped run where the
        masks of its clip scope let it (see shadow_pass.py)."""
        draws: dict[int, list[T]] = {}
        runs: dict[int, list[tuple[int, int]]] = {}
        for scope in (NO_CLIP, *range(len(self.clips))):
            chosen = sorted((pair for pair in casters if self.in_scope(pair[1], scope)), key=lambda pair: pair[1])
            draws[scope] = [draw for draw, _ in chosen]
            scope_runs: list[tuple[int, int]] = []
            for _, clip in chosen:
                if scope_runs and scope_runs[-1][0] == clip:
                    scope_runs[-1] = (clip, scope_runs[-1][1] + 1)
                else:
                    scope_runs.append((clip, 1))
            runs[scope] = scope_runs
        return draws, runs

    def mask_chain(self, clip: int) -> list[int]:
        """The clip scopes whose masks a point in ``clip`` must be inside,
        outermost first."""
        chain = []
        while clip != NO_CLIP:
            chain.append(clip)
            clip = self.clips[clip].parent
        return chain[::-1]


class SceneFlattener:
    """The first stage of the render chain: an RNode tree in, a
    :class:`FlatScene` out (see webgpu/packer.py for the whole chain).
    """

    def flatten(self, root: RNode) -> FlatScene:
        scene = FlatScene()
        _walk(root, np.eye(4), NO_CLIP, scene, mask_of=None, parent_alpha=1.0)
        return scene


def flatten_scene(root: RNode) -> FlatScene:
    """Flatten ``root`` once, without keeping anything for later calls."""
    return SceneFlattener().flatten(root)


def flatten(root: RNode) -> list[DrawItem]:
    """Depth-first list of visible leaves in painter's order (masks excluded)."""
    return flatten_scene(root).items


def bounding_box(root: RNode) -> tuple[np.ndarray, np.ndarray]:
    """World-space ``(min, max)`` corners over all anchors of all 2D leaves.

    Like Manim, only anchors count (not handles). Useful to center a group,
    e.g. Manim's ``move_to(ORIGIN)``.
    """
    anchors = [
        (item.world @ np.column_stack([anchors, np.zeros(len(anchors)), np.ones(len(anchors))]).T).T[:, :3]
        for item in flatten(root)
        if isinstance(item.obj, RShape2D)
        and len(anchors := item.obj.curves[:, [0, 3]].reshape(-1, 2))
    ]
    if not anchors:
        return np.zeros(3), np.zeros(3)
    points = np.concatenate(anchors)
    return points.min(axis=0), points.max(axis=0)


def _walk(
    obj: RNode,
    parent_world: np.ndarray,
    clip: int,
    scene: FlatScene,
    mask_of: ClipScope | None,
    parent_alpha: float,
) -> None:
    """Visit ``obj`` and its subtree.

    ``clip`` is the scope that clips whatever is drawn here. ``mask_of`` is
    set while walking a mask subtree: leaves then go into that scope's mask
    instead of the frame's draw list.
    """
    if not obj.visible:
        return
    # Child transforms compose with the parent: world = parent @ local.
    world = parent_world @ obj.transform.matrix()
    # Alphas multiply down the tree like transforms.
    alpha = max(0.0, parent_alpha * obj.alpha)

    if isinstance(obj, RLight):
        if mask_of is None:
            scene.lights.append(LightItem(obj, world, clip, alpha))
    elif isinstance(obj, RShape3D) and mask_of is not None:
        raise NotImplementedError("3D meshes cannot be part of a mask")
    elif isinstance(obj, (RBezierPath, RShape3D)):
        target = scene.items if mask_of is None else mask_of.mask
        target.append(DrawItem(obj, world, order=len(target), clip=clip, alpha=alpha))
    elif isinstance(obj, RGroup):
        for child in obj.children:
            _walk(child, world, clip, scene, mask_of, alpha)
    elif isinstance(obj, RClippingGroup):
        if mask_of is not None:
            raise NotImplementedError("an RClippingGroup inside a mask is not supported")
        content_clip = clip
        if obj.mask is not None:
            scope = ClipScope(obj, parent=clip)
            scene.clips.append(scope)
            content_clip = len(scene.clips) - 1
            _walk(obj.mask, world, clip, scene, mask_of=scope, parent_alpha=alpha)
        if obj.content is not None:
            _walk(obj.content, world, content_clip, scene, mask_of=None, parent_alpha=alpha)
    else:
        raise TypeError(f"unsupported render object: {type(obj).__name__}")
