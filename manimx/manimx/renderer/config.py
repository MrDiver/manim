"""Render settings and camera: the ``RenderConfig`` a renderer is given
next to the render object tree (manimx.robjects), and the views a frame
shows (``ViewState``)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..robjects import RNode
from ..robjects.transform import Transform, quaternion_from_rotation_matrix
from ..utils.color import BLACK, WHITE, ManimColor, ParsableManimColor, linear_rgb

DEFAULT_FRAME_HEIGHT = 8.0
"""Manim's default frame height in world units."""


@dataclass(frozen=True, slots=True)
class Camera:
    """A camera placed by a :class:`Transform`, looking along its local ``-z``.

    ``field_of_view=None`` gives Manim's default orthographic camera, which
    shows exactly ``frame_height`` world units vertically. A vertical field of
    view in radians switches to a perspective projection.

    Depth of field: things ``focus_distance`` away along the view axis are
    sharp; ``aperture`` is the blur of something infinitely far, as a
    fraction of the image height (0: everything sharp; see
    depth_of_field.wgsl).
    """

    transform: Transform = field(
        default_factory=lambda: Transform(position=(0.0, 0.0, 11.0))
    )
    frame_height: float = DEFAULT_FRAME_HEIGHT
    field_of_view: float | None = None
    near: float = 0.1
    far: float = 100.0
    focus_distance: float = 11.0
    aperture: float = 0.0

    @classmethod
    def look_at(
        cls,
        eye: tuple[float, float, float],
        target: tuple[float, float, float] = (0.0, 0.0, 0.0),
        up: tuple[float, float, float] = (0.0, 0.0, 1.0),
        **options: object,
    ) -> Camera:
        """Camera at ``eye`` looking at ``target``, with ``up`` pointing up on screen.

        The default ``up`` is world ``+z`` (Blender's and Manim's 3D
        convention); pass ``up=(0, 1, 0)`` for a camera above the XY plane.
        """
        eye_vector = np.asarray(eye, dtype=np.float64)
        forward = np.asarray(target, dtype=np.float64) - eye_vector
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-9:
            raise ValueError("camera up direction is parallel to the view direction")
        right /= np.linalg.norm(right)
        screen_up = np.cross(right, forward)
        # The camera looks along its local -z, so local +z points backwards.
        rotation = quaternion_from_rotation_matrix(np.column_stack([right, screen_up, -forward]))
        return cls(transform=Transform(position=eye, rotation=rotation), **options)  # type: ignore[arg-type]

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.transform.position)

    def view_matrix(self) -> np.ndarray:
        """World to view space (the inverse of the camera's placement)."""
        return np.linalg.inv(self.transform.matrix())

    def projection_matrix(self, aspect_ratio: float) -> np.ndarray:
        """View to clip space with WebGPU's depth range ``[0, 1]``."""
        near, far = self.near, self.far
        if self.field_of_view is None:
            half_height = self.frame_height / 2
            half_width = half_height * aspect_ratio
            return np.array(
                [
                    [1 / half_width, 0, 0, 0],
                    [0, 1 / half_height, 0, 0],
                    [0, 0, -1 / (far - near), -near / (far - near)],
                    [0, 0, 0, 1],
                ]
            )
        focal = 1 / math.tan(self.field_of_view / 2)
        return np.array(
            [
                [focal / aspect_ratio, 0, 0, 0],
                [0, focal, 0, 0],
                [0, 0, far / (near - far), near * far / (near - far)],
                [0, 0, -1, 0],
            ]
        )


MIN_SHADOW_TAPS = 4
"""The fewest taps a shadow lookup takes when renders share the taps."""


@dataclass(frozen=True, slots=True)
class RenderConfig:
    pixel_width: int = 1920
    pixel_height: int = 1080
    background_color: ParsableManimColor = BLACK
    background_opacity: float = 1.0
    camera: Camera = field(default_factory=Camera)
    msaa_samples: int = 4
    time: float = 0.0
    """Seconds, exposed to shaders as ``frame.time``."""
    layer_depth_step: float = 2e-7
    """Depth offset between consecutive 2D layers, see ``vector.wgsl``."""
    ambient_color: ParsableManimColor = WHITE
    ambient_intensity: float = 0.15
    """Constant ambient light on meshes, on top of the scene's lights."""
    exposure: float = 1.0
    """Multiplies lit mesh colors before tone mapping."""
    default_lights: bool = True
    """Light meshes with a key light when the scene contains no lights."""
    shadows: bool = True
    """Master switch for all shadows (lights and objects have their own)."""
    shadow_resolution: int = 2048
    """Width and height of every shadow map in texels."""
    shadow_softness: float = 1.0
    """Multiplies every light's ``size``: 0 gives hard shadows everywhere."""
    shadow_samples: int = 16
    """Taps per shadow lookup; more give smoother penumbras."""
    shadow_method: str = "pcss"
    """``"pcss"``: soft shadows whose edges widen with the distance to what
    casts them and the light's ``size`` (contact hardening). ``"pcf"``: a
    fixed narrow soft edge, much cheaper (previews)."""
    shadow_share: int = 1
    """How many renders a frame averages (``RenderSamples``). Each takes
    ``shadow_samples / shadow_share`` taps (at least MIN_SHADOW_TAPS), rotated
    differently per render, so the average still gets them all."""
    shadow_index: int = 0
    """Which of the ``shadow_share`` renders this is: it turns the tap
    pattern (the same in every frame, so unchanged frames stay equal)."""
    jitter: tuple[float, float] = (0.0, 0.0)
    """Moves the picture by this many pixels (right, up), for the renders
    of ``RenderSamples`` to cover each pixel."""
    camera_views: tuple[CameraViewState, ...] = ()
    """The camera views of the frame (see :class:`CameraViewState`): the
    shape with the uid of entry ``i`` shows layer ``i`` of the renderer's
    camera view texture."""
    hide_camera_views: bool = False
    """Leave the shapes of :attr:`camera_views` out: set while a camera
    view itself is rendered, so views never show views."""
    view_aspect: float | None = None
    """The width / height of the picture (default: of the pixels). A camera
    view renders at the frame's pixel size with the aspect of its plane."""

    @property
    def shadow_filter(self) -> tuple[float, float, float, float]:
        """``frame.shadow_filter``: the method (1 PCSS, 0 PCF), the taps per
        lookup and the turn (0 to 1) of the tap pattern for this render."""
        share = max(1, self.shadow_share)
        taps = min(self.shadow_samples, max(MIN_SHADOW_TAPS, math.ceil(self.shadow_samples / share)))
        # A golden-ratio sequence spreads the renders' patterns evenly.
        turn = (self.shadow_index * 0.6180339887) % 1.0
        return (1.0 if self.shadow_method == "pcss" else 0.0, float(taps), turn, 0.0)

    @property
    def aspect_ratio(self) -> float:
        if self.view_aspect is not None:
            return self.view_aspect
        return self.pixel_width / self.pixel_height

    @property
    def frame_width(self) -> float:
        return self.camera.frame_height * self.aspect_ratio

    @property
    def background_rgba(self) -> tuple[float, float, float, float]:
        r, g, b = ManimColor.parse(self.background_color).to_rgb()
        return (float(r), float(g), float(b), self.background_opacity)

    @property
    def ambient_linear(self) -> tuple[float, float, float]:
        """The ambient light in linear RGB, times its intensity."""
        r, g, b = np.array(linear_rgb(self.ambient_color)) * self.ambient_intensity
        return (float(r), float(g), float(b))


@dataclass(frozen=True, slots=True)
class CameraViewState:
    """What one camera view shows in a frame: the picture of ``camera``,
    drawn on the shape with ``uid`` (a ``CameraView`` node), which is
    ``width`` by ``height`` world units, centered on its local origin."""

    uid: int
    camera: Camera
    width: float
    height: float

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height


@dataclass(frozen=True, slots=True)
class ViewState:
    """One scene at one time together with the camera looking at it: what
    a timeline's or prepared scene's ``frame_at`` returns per active clip.
    A renderer draws ``root`` with a ``RenderConfig`` whose ``camera`` and
    ``camera_views`` are this view's. With motion blur, every sub-frame
    gets the camera of its own time, so camera moves blur too.
    """

    root: RNode
    """The render object tree at the time."""
    camera: Camera
    camera_views: tuple[CameraViewState, ...] = ()
    """The camera views in the tree, rendered before it (see
    :class:`CameraViewState`)."""


Frame = ViewState | Sequence[ViewState]
"""What a frame shows: one view, or several drawn over each other in
order (the active clips of a timeline, lowest layer first)."""
