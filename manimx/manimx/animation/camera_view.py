"""Camera views: what another camera sees, shown on a plane in the scene.

    side = Camera("side", position=(6, 0, 3))
    side.look_at(ORIGIN)
    screen = CameraView(side, 4, 2.25).move_to(UR * 2)
    self.add(screen)
    self.play(side.animate.shift(UP))       # the picture follows

A :class:`CameraView` is a Shape2D (a rectangle), so it moves, fades and
is masked by a ClippingGroup like any shape. Its fill is the picture of
its camera, multiplied by the fill color (white keeps it as it is) and
the fill opacity. The camera is an ordinary :class:`~.camera.Camera`
node: animate it like the scene's camera; it is never drawn itself.

Every frame, the renderer first renders each camera view of a frame at
the frame's pixel size and the aspect of its plane, then the frame.
Camera views are left out of those renders, so a view never shows a view
(no recursion).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any

import numpy as np

from ..renderer.config import CameraViewState
from .camera import Camera
from .node import Node
from .shape2d import Shape2D
from .signals import Signal


class CameraView(Shape2D):
    """A ``width`` by ``height`` rectangle showing what ``camera`` sees."""

    def __init__(self, camera: Camera, width: float = 4.0, height: float = 2.25, name: str | None = None, **style: Any) -> None:
        if not isinstance(camera, Camera):
            raise TypeError(f"CameraView needs a Camera, got {type(camera).__name__}")
        if width <= 0 or height <= 0:
            raise ValueError("a camera view needs a positive width and height")
        # The picture is the fill: shown fully, without an outline.
        style.setdefault("fill_opacity", 1.0)
        style.setdefault("stroke_width", 0.0)
        super().__init__(_rectangle(width, height), name, **style)
        self.camera = camera
        self.width = float(width)
        self.height = float(height)

    def view_state(self, value: Callable[[Signal], Any]) -> CameraViewState:
        """The renderer's description of this view, with signal values read by ``value``."""
        return CameraViewState(self.uid, self.camera.render_camera(value), self.width, self.height)


def current_value(signal: Signal) -> Any:
    """A signal's current value (what ``camera_views`` reads outside a plan)."""
    return signal.value


def camera_views(roots: Iterable[Node], value: Callable[[Signal], Any]) -> tuple[CameraViewState, ...]:
    """The views of every :class:`CameraView` under the present ``roots``
    that is not faded out (signal values read by ``value``)."""
    views: dict[int, CameraViewState] = {}
    for root in roots:
        if not value(root.present):
            continue
        for node in root.family():
            if isinstance(node, CameraView) and node.uid not in views and value(node.alpha) > 0.0:
                views[node.uid] = node.view_state(value)
    return tuple(views.values())


def view_cameras(roots: Iterable[Node]) -> Iterator[Camera]:
    """The cameras of every CameraView under ``roots`` (their signals are
    read to render the views)."""
    for root in roots:
        for node in root.family():
            if isinstance(node, CameraView):
                yield node.camera


def _rectangle(width: float, height: float) -> np.ndarray:
    """A closed rectangle centered on the origin, as straight cubic curves."""
    w, h = width / 2, height / 2
    corners = np.array([(w, h), (-w, h), (-w, -h), (w, -h), (w, h)])
    starts, ends = corners[:-1], corners[1:]
    return np.stack([starts, (2 * starts + ends) / 3, (starts + 2 * ends) / 3, ends], axis=1).reshape(-1, 2)


__all__ = ["CameraView", "camera_views", "current_value", "view_cameras"]
