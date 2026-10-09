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


def flatten_scene(root: RNode) -> FlatScene:
    scene = FlatScene()
    _walk(root, np.eye(4), NO_CLIP, scene, mask_of=None)
    return scene


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

    if isinstance(obj, RLight):
        if mask_of is None:
            scene.lights.append(LightItem(obj, world, clip))
    elif isinstance(obj, RShape3D) and mask_of is not None:
        raise NotImplementedError("3D meshes cannot be part of a mask")
    elif isinstance(obj, (RBezierPath, RShape3D)):
        target = scene.items if mask_of is None else mask_of.mask
        target.append(DrawItem(obj, world, order=len(target), clip=clip))
    elif isinstance(obj, RGroup):
        for child in obj.children:
            _walk(child, world, clip, scene, mask_of)
    elif isinstance(obj, RClippingGroup):
        if mask_of is not None:
            raise NotImplementedError("an RClippingGroup inside a mask is not supported")
        content_clip = clip
        if obj.mask is not None:
            scope = ClipScope(obj, parent=clip)
            scene.clips.append(scope)
            content_clip = len(scene.clips) - 1
            _walk(obj.mask, world, clip, scene, mask_of=scope)
        if obj.content is not None:
            _walk(obj.content, world, content_clip, scene, mask_of=None)
    else:
        raise TypeError(f"unsupported render object: {type(obj).__name__}")
