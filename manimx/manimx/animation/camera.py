"""Cameras: the node a scene is seen through.

Every scene has exactly one :class:`Camera`, ``Scene.camera``. Its
placement and lens are signals, so camera movement is animated with the
same ``play`` calls, rate functions, events and derived signals as any
content, and a clip's time map (start, cut, slow motion) moves the
camera together with everything else in the clip.

A camera is never drawn. For every active clip the timeline pairs the
clip's render state with its camera in a
:class:`~manimx.renderer.ViewState` (see ``Timeline.frame_at``).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

import numpy as np
import quaternion

from ..renderer.config import DEFAULT_FRAME_HEIGHT
from ..renderer.config import Camera as RenderCamera
from ..robjects import RNode, Transform
from . import authoring
from .kinds import DISCRETE, FACTOR, SCALAR
from .node import (
    ABS,
    REL,
    ChangeList,
    Node,
    PositionMixin,
    RotationMixin,
    _vector,
    animatable,
)
from .signals import RootSignal, Signal

DEFAULT_DISTANCE = 11.0
"""Manim's default camera distance from the z = 0 plane."""


class Camera(PositionMixin, RotationMixin, Node):
    """A camera at ``position``, looking along its local -z axis.

    * ``frame_height``: world units shown vertically (orthographic), a
      factor signal, so zooming in twice is uniform in speed.
    * ``field_of_view``: the vertical angle in radians, used while
      ``perspective`` is on (a discrete signal: projections cannot blend).
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        position: Sequence[float] = (0.0, 0.0, DEFAULT_DISTANCE),
        frame_height: float = DEFAULT_FRAME_HEIGHT,
        field_of_view: float | None = None,
        near: float = 0.1,
        far: float = 100.0,
        tags: Iterable[str] = (),
    ) -> None:
        super().__init__(name, tags=tags)
        self.position.set(_vector(position))
        self.frame_height = RootSignal(frame_height, FACTOR, name=f"{self.name}.frame_height")
        self.field_of_view = RootSignal(field_of_view or np.pi / 4, SCALAR, name=f"{self.name}.field_of_view")
        self.perspective = RootSignal(field_of_view is not None, DISCRETE, name=f"{self.name}.perspective")
        self.near = near
        self.far = far

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield from (self.frame_height, self.field_of_view, self.perspective)

    # -- changes ------------------------------------------------------------

    @animatable
    def look_at(self, target: Sequence[float] = (0.0, 0.0, 0.0), up: Sequence[float] = (0.0, 0.0, 1.0)) -> ChangeList:
        """Turn towards ``target`` (from where the camera is at this point of
        the scene). ``up`` points up on screen; looking along it, +y is used."""
        eye = np.array([authoring.current(axis) for axis in self.position.axes])
        return [(self.rotation, ABS, _look_rotation(eye, np.asarray(_vector(target)), np.asarray(_vector(up))))]

    @animatable
    def zoom(self, factor: float) -> ChangeList:
        """Show ``factor`` times less (orthographic): zoom(2) doubles sizes."""
        return [(self.frame_height, REL, 1.0 / factor)]

    @animatable
    def set_frame_height(self, height: float) -> ChangeList:
        return [(self.frame_height, ABS, height)]

    @animatable
    def set_field_of_view(self, angle: float) -> ChangeList:
        """Switch to a perspective projection with vertical angle ``angle``."""
        return [(self.perspective, ABS, True), (self.field_of_view, ABS, angle)]

    # -- render state -------------------------------------------------------

    def snapshot(self, alphas: tuple[Signal, ...] = ()) -> RNode | None:
        return None  # cameras are never drawn

    def render_camera(self) -> RenderCamera:
        """The renderer camera for the current signal values."""
        return RenderCamera(
            transform=Transform(position=self.position.value, rotation=self.rotation.value),
            frame_height=self.frame_height.value,
            field_of_view=self.field_of_view.value if self.perspective.value else None,
            near=self.near,
            far=self.far,
        )


def _look_rotation(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> quaternion.quaternion:
    forward = target - eye
    length = np.linalg.norm(forward)
    if length < 1e-12:
        raise ValueError("a camera cannot look at its own position")
    forward /= length
    if np.linalg.norm(np.cross(forward, up)) < 1e-9:
        up = np.array([0.0, 1.0, 0.0])  # looking along up: e.g. down onto the z = 0 plane
    return RenderCamera.look_at(tuple(eye), tuple(target), tuple(up)).transform.rotation

