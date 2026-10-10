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

Lights rarely move, so a light's records are cached by everything they
are computed from (its world matrix and settings, the bounds of its
scope, the config); only the frames' time is written per call.
:class:`ShadowLayers` goes further and tells which map layers would come
out exactly as before, so the renderer skips their passes.
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np

from ...robjects import (
    RDirectionalLight,
    RLight,
    RNode,
    RParticles,
    RPointLight,
    RShape3D,
    RSpotLight,
)
from ...robjects.rbezier_path import RBezierPath
from ...robjects.versioning import versioned_fields
from ..cache import RNodeDataCache
from ..config import RenderConfig
from ..scene_graph import NO_CLIP, FlatScene, LightItem
from .frame import FRAME_DTYPE, frame_data

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
    frames: np.ndarray  # FRAME_DTYPE records, one per map layer
    views: np.ndarray  # SHADOW_VIEW_DTYPE, one per map layer
    scopes: list[int]  # clip scope whose casters each layer renders
    changed: list[bool] | None = None
    """Per layer, whether it may differ from the same layer of the previous
    views of the same packer (None: render every layer)."""

    def __len__(self) -> int:
        return len(self.frames)


def build_shadow_views(
    scene: FlatScene,
    lights: list[LightItem],
    light_rows: np.ndarray,
    cache: RNodeDataCache,
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
        return ShadowViews(np.zeros(0, FRAME_DTYPE), np.zeros(0, SHADOW_VIEW_DTYPE), scopes)
    settings = _frame_settings(config)
    bounds_of: dict[int, tuple[np.ndarray, float] | None] = {}  # per scope: every light in it shares the bounds
    for row, item in zip(light_rows, lights):
        light = item.light
        if not light.cast_shadows:
            continue
        if item.clip not in bounds_of:
            bounds_of[item.clip] = scene_bounds(scene, cache, item.clip)
        bounds = bounds_of[item.clip]
        if bounds is None:
            continue
        row["shadow_view"] = len(scopes)
        key = (item.world.tobytes(), _light_settings(light), bounds[0].tobytes(), bounds[1], settings)
        light_frames, light_views = cache.get_or_create(
            light.uid, "shadow-views", key, lambda: _light_views(item, bounds, config)  # noqa: B023 - called right away
        )
        frames.append(light_frames)
        views.append(light_views)
        scopes.extend([item.clip] * len(light_views))
    if not scopes:
        return ShadowViews(np.zeros(0, FRAME_DTYPE), np.zeros(0, SHADOW_VIEW_DTYPE), scopes)
    all_frames = np.concatenate(frames)  # a copy: the cached records keep their time
    all_frames["time"] = config.time
    return ShadowViews(all_frames, np.concatenate(views), scopes)


def _light_views(item: LightItem, bounds: tuple[np.ndarray, float], config: RenderConfig) -> tuple[np.ndarray, np.ndarray]:
    """Frame and ShadowView records of one light's views."""
    light = item.light
    center, radius = bounds
    resolution = (config.shadow_resolution, config.shadow_resolution)
    frames: list[np.ndarray] = []
    views: list[np.ndarray] = []

    def add(view: np.ndarray, projection: np.ndarray, position: np.ndarray, params: tuple) -> None:
        frames.append(frame_data(view, projection, position, resolution, config))
        record = np.zeros((), SHADOW_VIEW_DTYPE)
        record["view_projection"] = (projection @ view).T
        record["params"] = params
        views.append(record)

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
    return np.array(frames, dtype=FRAME_DTYPE), np.array(views, dtype=SHADOW_VIEW_DTYPE)


def _light_settings(light: RLight) -> Hashable:
    """What a light's views depend on besides its world matrix."""
    if isinstance(light, RSpotLight):
        return (type(light), light.range, light.outer_cone_angle)
    if isinstance(light, RPointLight):
        return (type(light), light.range)
    return type(light)


def _frame_settings(config: RenderConfig) -> Hashable:
    """The config values a Frame record holds, except the time."""
    return (
        config.shadow_resolution,
        config.layer_depth_step,
        config.exposure,
        config.ambient_linear,
        config.shadow_softness,
        config.shadow_samples,
        config.shadows,
    )


def scene_bounds(scene: FlatScene, cache: RNodeDataCache, scope: int = NO_CLIP) -> tuple[np.ndarray, float] | None:
    """Bounding sphere ``(center, radius)`` of the drawn objects in ``scope``
    (all objects for NO_CLIP), or None if there are none."""
    boxes: list[np.ndarray] = []
    worlds: list[np.ndarray] = []
    for item in scene.items:
        obj = item.obj
        if not scene.in_scope(item.clip, scope):
            continue
        if isinstance(obj, RParticles) and obj.mesh.vertex_count:
            local = _particle_box(obj, cache)
            if local is None:
                continue
        elif isinstance(obj, RShape3D) and obj.mesh.vertex_count:
            local = cache.get_or_create(obj.uid, "bounds", obj.version("mesh"), lambda: _box(obj.mesh.positions))
        elif isinstance(obj, RBezierPath) and obj.num_curves:
            local = cache.get_or_create(obj.uid, "bounds", obj.version("points"), lambda: _box(obj.points_3d()))
        else:
            continue
        boxes.append(local)
        worlds.append(item.world)
    if not boxes:
        return None
    # All boxes' corners to world space in one go.
    world = np.stack(worlds)
    corners = np.einsum("nij,nkj->nki", world[:, :3, :3], np.stack(boxes)) + world[:, None, :3, 3]
    points = corners.reshape(-1, 3)
    low, high = points.min(axis=0), points.max(axis=0)
    return (low + high) / 2, max(float(np.linalg.norm(high - low)) / 2, 1e-3)


def _particle_box(obj: RParticles, cache: RNodeDataCache) -> np.ndarray | None:
    """The corners of a box around every shown particle: their positions,
    grown by the mesh's reach times the largest scale."""
    shown = np.all(np.abs(obj.scales) > 1e-9, axis=1)
    if not shown.any():
        return None
    mesh = cache.get_or_create(obj.uid, "bounds", obj.version("mesh"), lambda: _box(obj.mesh.positions))
    reach = float(np.linalg.norm(mesh, axis=1).max() * np.abs(obj.scales[shown]).max())
    positions = obj.positions[shown]
    return _box(np.vstack([positions.min(axis=0) - reach, positions.max(axis=0) + reach]))


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


class ShadowLayers:
    """Tells which shadow map layers would come out exactly as before.

    A layer's depth depends on its view (the Frame record it is rendered
    with) and on the shadow casters of its clip scope: their geometry,
    their style (the shadow pass discards transparent fragments) and their
    world matrix. Render nodes are immutable, so an object identity plus
    its version tokens and world matrix stand for all of that. The
    objects of the previous call are held, so their ids cannot be reused
    by new objects while they are compared.
    """

    def __init__(self) -> None:
        self._keys: list[Hashable] = []
        self._held: list[object] = []

    def mark(self, scene: FlatScene, shadows: ShadowViews) -> None:
        """Set ``shadows.changed`` relative to the previous call."""
        held: list[object] = []
        casters: dict[int, Hashable] = {}
        frames = shadows.frames.copy()
        frames["time"] = 0.0  # no shader reads it in shadow passes
        keys: list[Hashable] = []
        for index, scope in enumerate(shadows.scopes):
            if scope not in casters:
                casters[scope] = _casters(scene, scope, held)
            keys.append((frames[index].tobytes(), casters[scope]))
        previous = self._keys
        shadows.changed = [index >= len(previous) or key != previous[index] for index, key in enumerate(keys)]
        self._keys, self._held = keys, held


def _casters(scene: FlatScene, scope: int, held: list[object]) -> Hashable:
    """Identity, versions and world matrix of every shadow caster in ``scope``,
    and of the mask shapes that clip them (they shape the shadow too)."""
    key = []
    masks: set[int] = set()
    for item in scene.items:
        obj = item.obj
        if not obj.cast_shadows or not scene.in_scope(item.clip, scope):
            continue
        masks.update(scene.mask_chain(item.clip))
        held.append(obj)
        part: tuple = (id(obj), _versions(obj), item.world.tobytes())
        if isinstance(obj, RParticles):
            # Particle arrays are not versioned: hold them and use their ids.
            arrays = (obj.positions, obj.rotations, obj.scales, obj.colors)
            held.extend(arrays)
            part += tuple(map(id, arrays))
        key.append(part)
    for clip in sorted(masks):
        for item in scene.clips[clip].mask:
            held.append(item.obj)
            key.append((id(item.obj), _versions(item.obj), item.world.tobytes()))
    return tuple(key)


def _versions(obj: RNode) -> tuple[Hashable, ...]:
    return obj.version(*sorted(versioned_fields(type(obj))))
