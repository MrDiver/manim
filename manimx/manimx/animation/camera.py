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

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any

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


def _current(signal: Signal) -> Any:
    return signal.value


DEFAULT_DISTANCE = 11.0
"""Manim's default camera distance from the z = 0 plane."""


class Camera(PositionMixin, RotationMixin, Node):
    """A camera at ``position``, looking along its local -z axis.

    * ``frame_height``: world units shown vertically (orthographic), a
      factor signal, so zooming in twice is uniform in speed.
    * ``field_of_view``: the vertical angle in radians, used while
      ``perspective`` is on (a discrete signal: projections cannot blend).
    * ``focus_distance`` and ``aperture``: depth of field. Things that far
      along the view axis are sharp; ``aperture`` is the blur of something
      infinitely far, as a fraction of the image height (0: no blur).
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
        focus_distance: float = DEFAULT_DISTANCE,
        aperture: float = 0.0,
        tags: Iterable[str] = (),
    ) -> None:
        super().__init__(name, tags=tags)
        self.position.set(_vector(position))
        self.frame_height = RootSignal(frame_height, FACTOR, name=f"{self.name}.frame_height")
        self.field_of_view = RootSignal(field_of_view or np.pi / 4, SCALAR, name=f"{self.name}.field_of_view")
        self.perspective = RootSignal(field_of_view is not None, DISCRETE, name=f"{self.name}.perspective")
        self.focus_distance = RootSignal(focus_distance, SCALAR, name=f"{self.name}.focus_distance")
        self.aperture = RootSignal(aperture, SCALAR, name=f"{self.name}.aperture")
        self.near = near
        self.far = far

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield from (self.frame_height, self.field_of_view, self.perspective, self.focus_distance, self.aperture)

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

    @animatable
    def set_focus(self, distance: float) -> ChangeList:
        """Focus on things ``distance`` away along the view axis."""
        if distance <= 0:
            raise ValueError("a focus distance must be positive")
        return [(self.focus_distance, ABS, float(distance))]

    @animatable
    def focus_on(self, point: Sequence[float]) -> ChangeList:
        """Focus on ``point`` (from where the camera is and looks at this
        point of the scene)."""
        eye = np.array([authoring.current(axis) for axis in self.position.axes])
        looking = -quaternion.as_rotation_matrix(authoring.current(self.rotation))[:, 2]
        distance = float(np.dot(np.asarray(_vector(point)) - eye, looking))
        if distance <= 0:
            raise ValueError("the camera can only focus on points in front of it")
        return [(self.focus_distance, ABS, distance)]

    @animatable
    def set_aperture(self, aperture: float) -> ChangeList:
        """Blur things out of focus: ``aperture`` is the blur of something
        infinitely far, as a fraction of the image height (0: no blur)."""
        if aperture < 0:
            raise ValueError("an aperture cannot be negative")
        return [(self.aperture, ABS, float(aperture))]

    # -- render state -------------------------------------------------------

    def snapshot(self) -> RNode | None:
        return None  # cameras are never drawn

    def render_camera(self, value: Callable[[Signal], Any] = _current) -> RenderCamera:
        """The renderer camera for the signal values ``value`` reads (by
        default the current ones; a ScenePlan reads its frame state)."""
        return RenderCamera(
            transform=Transform(position=tuple(value(axis) for axis in self.position.axes), rotation=value(self.rotation)),
            frame_height=value(self.frame_height),
            field_of_view=value(self.field_of_view) if value(self.perspective) else None,
            near=self.near,
            far=self.far,
            focus_distance=value(self.focus_distance),
            aperture=max(0.0, value(self.aperture)),
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

