"""The renderer interface: the ``RenderConfig`` and ``Camera`` a renderer is
given next to the render object tree (``RNode`` and its subclasses in
``manim._native``), and the drawing methods used by Manager."""

from __future__ import annotations

import enum
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from manim.utils.color import BLACK, ManimColor, ParsableManimColor
from manim.utils.transform import Transform

if TYPE_CHECKING:
    from manim.mobject.mobject import Mobject
    from manim._native import RNode
    from manim.scene.scene import Scene
    from manim.typing import RGBAPixelArray

__all__ = [
    "Camera",
    "CameraViewState",
    "CapabilityException",
    "RenderConfig",
    "Renderer",
    "ShadowMethod",
]

DEFAULT_FRAME_HEIGHT = 8.0
"""Manim's default frame height in world units."""


@dataclass(frozen=True, slots=True)
class Camera:
    """A camera placed by a :class:`Transform`, looking along its local ``-z``.

    The snapshot of the scene's camera at one time: plain values only, made
    by the scene side and read by the renderer.

    ``field_of_view=None`` gives Manim's default orthographic camera, which
    shows exactly ``frame_height`` world units vertically. A vertical field of
    view in radians switches to a perspective projection.

    Depth of field: things ``focus_distance`` away along the view axis are
    sharp; ``aperture`` is the blur of something infinitely far, as a
    fraction of the image height (0: everything sharp).
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


class ShadowMethod(enum.Enum):
    """How shadow edges are softened."""

    PCSS = "pcss"
    """Soft shadows whose edges widen with the distance to what casts them
    and the light's ``size`` (contact hardening)."""
    PCF = "pcf"
    """A fixed narrow soft edge, much cheaper (previews)."""


@dataclass(frozen=True, slots=True)
class RenderConfig:
    """What a renderer is given next to the render snapshots of one frame."""

    # Not ported yet: every field typed ``... | None = None`` with a
    # ``# manimx:`` comment below. They are optional only until a renderer
    # implements them; as each is ported, give it back its manimx type and
    # default (the comment) and drop it from this list: gpu, msaa_samples,
    # time, layer_depth_step, ambient_color, ambient_intensity, exposure,
    # default_lights, shadows, shadow_resolution, shadow_softness,
    # shadow_samples, shadow_method, shadow_share, shadow_index, jitter,
    # camera_views, hide_camera_views. ``view_aspect`` is optional in
    # manimx itself.
    pixel_width: int = 1920
    pixel_height: int = 1080
    background_color: ParsableManimColor = BLACK
    background_opacity: float = 1.0
    camera: Camera = field(default_factory=Camera)
    gpu: str | None = None  # manimx: "high-performance"
    """Which GPU renders: ``"high-performance"`` a dedicated one,
    ``"low-power"`` an integrated one (when the dedicated one is busy, or
    to save power); the only one when there is just one."""
    msaa_samples: int | None = None  # manimx: 4
    time: float | None = None  # manimx: 0.0
    """Seconds, exposed to shaders as ``frame.time``."""
    layer_depth_step: float | None = None  # manimx: 2e-7
    """Depth offset between consecutive 2D layers."""
    ambient_color: ParsableManimColor | None = None  # manimx: WHITE
    ambient_intensity: float | None = None  # manimx: 0.15
    """Constant ambient light on meshes, on top of the scene's lights."""
    exposure: float | None = None  # manimx: 1.0
    """Multiplies lit mesh colors before tone mapping."""
    default_lights: bool | None = None  # manimx: True
    """Light meshes with a key light when the scene contains no lights."""
    shadows: bool | None = None  # manimx: True
    """Master switch for all shadows (lights and objects have their own)."""
    shadow_resolution: int | None = None  # manimx: 2048
    """Width and height of every shadow map in texels."""
    shadow_softness: float | None = None  # manimx: 1.0
    """Multiplies every light's ``size``: 0 gives hard shadows everywhere."""
    shadow_samples: int | None = None  # manimx: 16
    """Taps per shadow lookup; more give smoother penumbras."""
    shadow_method: ShadowMethod | None = None  # manimx: "pcss" (here ShadowMethod.PCSS)
    shadow_share: int | None = None  # manimx: 1
    """How many renders a frame averages. Each takes a share of
    ``shadow_samples`` taps, rotated differently per render, so the
    average still gets them all."""
    shadow_index: int | None = None  # manimx: 0
    """Which of the ``shadow_share`` renders this is: it turns the tap
    pattern (the same in every frame, so unchanged frames stay equal)."""
    jitter: tuple[float, float] | None = None  # manimx: (0.0, 0.0)
    """Moves the picture by this many pixels (right, up), for the renders
    of a frame to cover each pixel."""
    camera_views: tuple[CameraViewState, ...] | None = None  # manimx: ()
    """The camera views of the frame (see :class:`CameraViewState`): the
    shape with the uid of entry ``i`` shows layer ``i`` of the renderer's
    camera view texture."""
    hide_camera_views: bool | None = None  # manimx: False
    """Leave the shapes of :attr:`camera_views` out: set while a camera
    view itself is rendered, so views never show views."""
    view_aspect: float | None = None
    """The width / height of the picture (default: of the pixels). A camera
    view renders at the frame's pixel size with the aspect of its plane."""

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


@dataclass(frozen=True, slots=True)
class CameraViewState:
    """What one camera view shows in a frame: the picture of ``camera``,
    drawn on the shape with ``uid``, which is ``width`` by ``height`` world
    units, centered on its local origin."""

    uid: int
    camera: Camera
    width: float
    height: float

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height


class Renderer(Protocol):
    """Draw the render snapshots of one frame into an image.

    A renderer is a long-lived service. It is given the snapshots and a
    :class:`RenderConfig`, whose ``camera`` is the snapshot of the scene's
    camera at that time, and nothing else: it never reads a Scene, a
    Mobject or the animation clock. It may keep what it built between
    frames, keyed by ``RNode.uid`` and the versions of its values
    (``RNode.version(field)``).
    """

    def render(
        self, nodes: Sequence[RNode], config: RenderConfig, /
    ) -> RGBAPixelArray:
        """Draw ``nodes`` in order with ``config``; rows top to bottom."""
        ...


class CapabilityException(Exception):
    """Raised by a renderer when it is asked for something it cannot do.

    The message says what was asked for and why the renderer cannot do it.
    """


class _AnimationRenderer(Protocol):
    """Draw frames and display them in a preview window at Manager's request.

    Manager runs animations and sends frames to the writer or returns a requested
    frame. The renderer draws the images and paces the live preview.
    """

    def _animation_cache_identity(self, scene: Scene) -> tuple[str, Any]: ...

    def _start_animation(self) -> None: ...

    def _prepare_animation(self, scene: Scene) -> None: ...

    def render(
        self, scene: Scene, frame_offset: float, moving_mobjects: list[Mobject], /
    ) -> None: ...

    def get_frame(self) -> RGBAPixelArray: ...

    def _present_frame(self, scene: Scene, frame_offset: float) -> None: ...

    def _present_frozen_frame(self, scene: Scene, duration: float) -> None: ...
