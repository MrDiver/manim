"""Where shadow maps are rendered from (pure numpy, like the batch modules).

Every shadow-casting light gets one or more *shadow views*: a camera placed
at the light that renders the depth of all casters into one layer of the
shadow map array.

    directional  1 view, orthographic, fitted around the scene's bounds
    spot         1 view, perspective, covering the light's outer cone
    point        6 views, perspective 90 degrees, one per cube face
                 (+x, -x, +y, -y, +z, -z; the shader picks the face from
                 the direction to the light)

Each view yields a Frame record (to render the map with the normal
pipelines) and a ShadowView record (to look the map up when lighting).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...objects import RDirectionalLight, RPointLight, RShape3D, RSpotLight
from ...objects.rbezier_path import RBezierPath
from ..cache import MobjectDataCache
from ..config import RenderConfig
from ..scene_graph import NO_CLIP, FlatScene, LightItem
from .frame import frame_data

# Mirrors `struct ShadowView` in shaders/shadows.wgsl.
SHADOW_VIEW_DTYPE = np.dtype(
    [
        ("view_projection", "<f4", (4, 4)),
        # near, far, frustum width (world units for orthographic views, at
        # distance 1 for perspective ones), 1 = perspective / 0 = orthographic
        ("params", "<f4", 4),
    ]
)
assert SHADOW_VIEW_DTYPE.itemsize == 80

NO_SHADOW = -1
_NEAR = 0.05
# Cube faces: (looking along, up). Order matches face_index() in shadows.wgsl.
_CUBE_FACES = (
    ((1, 0, 0), (0, 0, 1)),
    ((-1, 0, 0), (0, 0, 1)),
    ((0, 1, 0), (0, 0, 1)),
    ((0, -1, 0), (0, 0, 1)),
    ((0, 0, 1), (0, 1, 0)),
    ((0, 0, -1), (0, 1, 0)),
)


@dataclass(slots=True)
class ShadowViews:
    frames: list[np.ndarray]  # FRAME_DTYPE records, one per map layer
    views: np.ndarray  # SHADOW_VIEW_DTYPE, one per map layer
    scopes: list[int]  # clip scope whose casters each layer renders

    def __len__(self) -> int:
        return len(self.frames)


def build_shadow_views(
    scene: FlatScene,
    lights: list[LightItem],
    light_rows: np.ndarray,
    cache: MobjectDataCache,
    config: RenderConfig,
) -> ShadowViews:
    """Assign shadow map layers to lights; writes ``light_rows["shadow_view"]``.

    ``lights`` and ``light_rows`` must correspond one to one.
    """
    light_rows["shadow_view"] = NO_SHADOW
    frames: list[np.ndarray] = []
    views: list[np.ndarray] = []
    scopes: list[int] = []
    if not config.shadows:
        return ShadowViews(frames, np.zeros(0, SHADOW_VIEW_DTYPE), scopes)
    resolution = (config.shadow_resolution, config.shadow_resolution)
    scope = NO_CLIP

    def add(view: np.ndarray, projection: np.ndarray, position: np.ndarray, params: tuple) -> None:
        scopes.append(scope)
        frames.append(frame_data(view, projection, position, resolution, config))
        record = np.zeros((), SHADOW_VIEW_DTYPE)
        record["view_projection"] = (projection @ view).T
        record["params"] = params
        views.append(record)

    for row, item in zip(light_rows, lights):
        light = item.light
        bounds = scene_bounds(scene, cache, item.clip)
        if not light.cast_shadows or bounds is None:
            continue
        center, radius = bounds
        scope = item.clip
        row["shadow_view"] = len(frames)
        position = item.world[:3, 3]
        direction = -item.world[:3, 2] / np.linalg.norm(item.world[:3, 2])
        if isinstance(light, RDirectionalLight):
            # Look at the scene's bounding sphere from outside it.
            eye = center - direction * 2 * radius
            near, far = radius * 0.5, radius * 3.5
            add(_look(eye, direction), _orthographic(radius, near, far), eye, (near, far, 2 * radius, 0.0))
        elif isinstance(light, RSpotLight):
            far = light.range or float(np.linalg.norm(position - center) + radius)
            fov = min(2 * light.outer_cone_angle + 0.05, math.radians(170))
            add(_look(position, direction), _perspective(fov, _NEAR, far), position, (_NEAR, far, 2 * math.tan(fov / 2), 1.0))
        elif isinstance(light, RPointLight):
            far = light.range or float(np.linalg.norm(position - center) + radius)
            for forward, up in _CUBE_FACES:
                add(_look(position, np.array(forward, float), np.array(up, float)), _perspective(math.pi / 2, _NEAR, far), position, (_NEAR, far, 2.0, 1.0))
    return ShadowViews(frames, np.array(views, dtype=SHADOW_VIEW_DTYPE), scopes)


def scene_bounds(scene: FlatScene, cache: MobjectDataCache, scope: int = NO_CLIP) -> tuple[np.ndarray, float] | None:
    """Bounding sphere ``(center, radius)`` of the drawn objects in ``scope``
    (all objects for NO_CLIP), or None if there are none."""
    corners = []
    for item in scene.items:
        obj = item.obj
        if not scene.in_scope(item.clip, scope):
            continue
        if isinstance(obj, RShape3D):
            local = cache.get_or_create(obj.uid, "bounds", obj.version("mesh"), lambda: _box(obj.mesh.positions))
        elif isinstance(obj, RBezierPath) and obj.num_curves:
            local = cache.get_or_create(obj.uid, "bounds", obj.version("points"), lambda: _box(obj.points_3d()))
        else:
            continue
        corners.append(local @ item.world[:3, :3].T + item.world[:3, 3])
    if not corners:
        return None
    points = np.concatenate(corners)
    low, high = points.min(axis=0), points.max(axis=0)
    return (low + high) / 2, max(float(np.linalg.norm(high - low)) / 2, 1e-3)


def _box(points: np.ndarray) -> np.ndarray:
    """The 8 corners of the axis-aligned box around ``points``."""
    low, high = points.min(axis=0), points.max(axis=0)
    return np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])


def _look(eye: np.ndarray, forward: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """View matrix of a camera at ``eye`` looking along ``forward``."""
    if up is None:
        up = np.array([0.0, 0.0, 1.0]) if abs(forward[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    screen_up = np.cross(right, forward)
    rotation = np.stack([right, screen_up, -forward])  # rows: camera axes
    view = np.eye(4)
    view[:3, :3] = rotation
    view[:3, 3] = -rotation @ eye
    return view


def _orthographic(half_size: float, near: float, far: float) -> np.ndarray:
    return np.array(
        [
            [1 / half_size, 0, 0, 0],
            [0, 1 / half_size, 0, 0],
            [0, 0, -1 / (far - near), -near / (far - near)],
            [0, 0, 0, 1],
        ]
    )


def _perspective(fov: float, near: float, far: float) -> np.ndarray:
    focal = 1 / math.tan(fov / 2)
    return np.array(
        [
            [focal, 0, 0, 0],
            [0, focal, 0, 0],
            [0, 0, far / (near - far), near * far / (near - far)],
            [0, 0, -1, 0],
        ]
    )
