"""The renderer interface: the ``RenderConfig`` and ``Camera`` a renderer is
given next to the render object tree (``RNode`` and its subclasses in
``manim._native``), and the drawing methods used by Manager.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from manim.utils.color import BLACK, WHITE, ManimColor, ParsableManimColor
from manim.utils.transform import Transform

if TYPE_CHECKING:
    from manim._native import RNode
    from manim.mobject.mobject import Mobject
    from manim.scene.scene import Scene
    from manim.typing import RGBAPixelArray

__all__ = [
    "Camera",
    "CameraViewState",
    "CapabilityException",
    "RenderConfig",
    "Renderer",
    "ShadowMethod",
    "ToneMapOperator",
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


class ToneMapOperator(enum.Enum):
    """How lit 3D colors (HDR radiance) are mapped to the screen. Values
    match the TONE_MAP_* constants in ``native/shaders/color.wgsl``.
    """

    NONE = 0
    """Clamp to [0, 1]."""
    PBR_NEUTRAL = 1
    """Khronos PBR Neutral: base colors stay faithful (the default)."""
    ACES = 2
    """Filmic ACES fit: strong contrast, desaturated highlights."""
    AGX = 3
    """AgX, Blender's default: smooth highlight roll-off."""
    REINHARD = 4
    """x / (1 + x): simple, low contrast."""


class ShadowMethod(enum.Enum):
    """How shadow edges are softened."""

    PCSS = "pcss"
    """Soft shadows whose edges widen with the distance to what casts them
    and the light's ``size`` (contact hardening)."""
    PCF = "pcf"
    """A fixed narrow soft edge, much cheaper (previews)."""


@dataclass(frozen=True, slots=True)
class RenderConfig:
    """What a renderer is given next to the render object tree of one frame."""

    pixel_width: int = 1920
    pixel_height: int = 1080
    background_color: ParsableManimColor = BLACK
    background_opacity: float = 1.0
    camera: Camera = field(default_factory=Camera)
    gpu: str = "high-performance"
    """Which GPU renders: ``"high-performance"`` a dedicated one,
    ``"low-power"`` an integrated one (when the dedicated one is busy, or
    to save power); the only one when there is just one."""
    gpu_memory: str = "compact"
    """How the GPU memory is reserved: ``"compact"`` in small blocks, so
    many renderers and processes (parallel tests, render workers) fit on
    one GPU; ``"fast"`` in large blocks, with fewer allocations. A process
    keeps the blocks it reserved, so try ``"compact"`` when the GPU runs
    out of memory."""
    msaa_samples: int = 4
    """Samples per pixel (multisample anti-aliasing): 1 and 4 work on every
    GPU, 2, 8 and 16 where the GPU has them; -1 takes the most it has. More
    samples smooth edges further and cost render time and memory."""
    time: float = 0.0
    """Seconds, exposed to shaders as ``frame.time``."""
    layer_depth_step: float = 2e-7
    """Depth offset between consecutive 2D layers."""
    ambient_color: ParsableManimColor = WHITE
    ambient_intensity: float = 0.15
    """Constant ambient light on meshes, on top of the scene's lights."""
    exposure: float = 1.0
    """Multiplies lit mesh colors before tone mapping."""
    tone_map: ToneMapOperator = ToneMapOperator.PBR_NEUTRAL
    """How lit mesh colors are mapped to the screen."""
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
    shadow_method: ShadowMethod = ShadowMethod.PCSS
    shadow_share: int = 1
    """How many renders a frame averages. Each takes a share of
    ``shadow_samples`` taps, rotated differently per render, so the
    average still gets them all."""
    shadow_index: int = 0
    """Which of the ``shadow_share`` renders this is: it turns the tap
    pattern (the same in every frame, so unchanged frames stay equal)."""
    jitter: tuple[float, float] = (0.0, 0.0)
    """Moves the picture by this many pixels (right, up), for the renders
    of a frame to cover each pixel."""
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
    def ambient_rgb(self) -> tuple[float, float, float]:
        """The ambient color as sRGB floats, as a renderer reads it."""
        r, g, b = ManimColor.parse(self.ambient_color).to_rgb()
        return (float(r), float(g), float(b))


@dataclass(frozen=True, slots=True)
class CameraViewState:
    """What one camera view shows in a frame: the picture of ``camera``,
    drawn on the shape with ``uid``, which is ``width`` by ``height`` world
    units, centered on its local origin.
    """

    uid: int
    camera: Camera
    width: float
    height: float

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height


class Renderer(Protocol):
    """Draw the render object tree of one frame into an image.

    A renderer is a long-lived service. It is given the root of the tree
    and a :class:`RenderConfig`, whose ``camera`` is the snapshot of the
    scene's camera at that time, and nothing else: it never reads a Scene,
    a Mobject or the animation clock. It may keep what it built between
    frames, keyed by ``RNode.uid`` and the versions of its values
    (``RNode.version(field)``).
    """

    def render(self, root: RNode, config: RenderConfig, /) -> RGBAPixelArray:
        """Draw the tree under ``root`` with ``config``; rows top to bottom."""
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
