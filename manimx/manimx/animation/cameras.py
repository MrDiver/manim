"""Cameras: mobjects that say where a view looks from (section 8.1).

A :class:`Camera` is a mobject like any other: its placement and lens are
signals, so camera movement is animated with the same ``play`` calls,
rate functions, events and time maps as any content. A scene that only
moves a camera is a *camera scene*; a camera can also live in a content
scene and move with it.

A camera is never drawn. A :class:`View` pairs a world with a camera, and
the timeline turns it into a :class:`~manimx.renderer.ViewState` (the
world's render state plus the renderer camera) at any time.

Every signal belongs to the one scene that animates it, so a camera is
moved by exactly one scene. To cut between camera moves of different
scenes, use one camera per scene and switch views.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import quaternion

from ..objects import RNode, Transform
from ..renderer.config import DEFAULT_FRAME_HEIGHT
from ..renderer.config import Camera as RenderCamera
from . import authoring
from .kinds import DISCRETE, FACTOR, SCALAR
from .mobjects import ABS, REL, ChangeList, Mobject, _vector, animatable
from .signals import RootSignal, Signal

if TYPE_CHECKING:
    from .worlds import World

DEFAULT_DISTANCE = 11.0
"""Manim's default camera distance from the z = 0 plane."""


class Camera(Mobject):
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
    ) -> None:
        super().__init__(name)
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

    def render_camera(self, parent: Transform | None = None) -> RenderCamera:
        """The renderer camera for the current signal values, placed inside
        ``parent`` (the transform of the clip the camera plays in)."""
        transform = Transform(position=self.position.value, rotation=self.rotation.value)
        if parent is not None:
            transform = _compose(parent, transform)
        return RenderCamera(
            transform=transform,
            frame_height=self.frame_height.value * (1.0 if parent is None else parent.scale[1]),
            field_of_view=self.field_of_view.value if self.perspective.value else None,
            near=self.near,
            far=self.far,
        )


@dataclass(eq=False)
class View:
    """A world seen through a camera (the image a composition shows)."""

    world: World
    camera: Camera
    name: str = "view"


def _look_rotation(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> quaternion.quaternion:
    forward = target - eye
    length = np.linalg.norm(forward)
    if length < 1e-12:
        raise ValueError("a camera cannot look at its own position")
    forward /= length
    if np.linalg.norm(np.cross(forward, up)) < 1e-9:
        up = np.array([0.0, 1.0, 0.0])  # looking along up: e.g. down onto the z = 0 plane
    return RenderCamera.look_at(tuple(eye), tuple(target), tuple(up)).transform.rotation


def _compose(parent: Transform, child: Transform) -> Transform:
    """``parent`` applied after ``child``, for rigid children (no scale)."""
    position = parent.apply([child.position])[0]
    return Transform(position=tuple(position), rotation=parent.rotation * child.rotation)
