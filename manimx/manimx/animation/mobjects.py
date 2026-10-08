"""Scene objects: the authoring-side objects whose properties are signals.

A :class:`Mobject` is what scene code creates and animates. Every
animatable property is a signal (positions and scales per axis), and
:meth:`Mobject.snapshot` turns the current signal values into render
state (manimx.objects) for the renderer. Snapshots reuse the mobject's
``uid`` and carry version tokens from the signals, so renderer caches
survive any number of snapshots (section 9 of the design doc).

Change methods (``shift``, ``move_to``, ``set_fill``, ...) are defined once
with :func:`animatable`, as a function returning ``(signal, change,
value)`` triples. Called directly, the changes go through the authoring
context (authoring.py); called on ``mobject.animate``, they are collected
into an animation. Both paths share one definition.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import quaternion

from ..objects import ClippingGroup as RenderClippingGroup
from ..objects import Fill, Shape2D, Stroke, Transform
from ..objects import Group as RenderGroup
from ..objects import RenderObject, new_uid
from ..objects.geometry import arc_points, regular_polygon, rectangle
from ..objects.transform import quaternion_from_axis_angle
from ..utils.color import WHITE, ManimColor, ParsableManimColor
from . import authoring
from .animations import AnimationBuilder
from .kinds import COLOR, DISCRETE, FACTOR, PATH, ROTATION, SCALAR
from .signals import RootSignal, Signal, Vector3Signal, derive
from .snapshot import version_token
from .tracks import Change

ChangeList = list[tuple[RootSignal, Change, Any]]

REL, ABS = Change.RELATIVE, Change.ABSOLUTE


def animatable(changes: Callable[..., ChangeList]) -> Callable[..., Any]:
    """Turn a change definition into a method usable directly and via
    ``.animate`` (see the module docs)."""

    @functools.wraps(changes)
    def method(self: Mobject, *args: Any, **kwargs: Any) -> Mobject:
        for signal, mode, value in changes(self, *args, **kwargs):
            authoring.change(signal, mode, value)
        return self

    method._signal_changes = changes  # type: ignore[attr-defined]
    return method


def _rgb(color: ParsableManimColor) -> tuple[float, float, float]:
    r, g, b = ManimColor.parse(color).to_rgb()
    return (float(r), float(g), float(b))


class Mobject:
    """Something in a scene: identity, a transform made of signals,
    an opacity multiplier and presence.

    * ``position``, ``scale``: one signal per axis, so animations of
      different axes never interfere.
    * ``rotation``: a quaternion signal.
    * ``alpha``: an opacity multiplier for fades; it multiplies the fill
      and stroke opacity of the whole subtree.
    * ``present``: whether the mobject is in the scene; managed by the
      scene's ``add`` and ``remove``.
    """

    def __init__(self, name: str | None = None) -> None:
        self.uid = new_uid()
        self.name = name or f"{type(self).__name__.lower()}-{self.uid}"
        self.position = Vector3Signal(name=f"{self.name}.position")
        self.rotation = RootSignal(quaternion.one, ROTATION, name=f"{self.name}.rotation")
        self.scale = Vector3Signal((1.0, 1.0, 1.0), name=f"{self.name}.scale", kind=FACTOR)
        self.alpha = RootSignal(1.0, SCALAR, name=f"{self.name}.alpha")
        self.present = RootSignal(False, DISCRETE, name=f"{self.name}.present")
        self.children: list[Mobject] = []

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"

    # -- structure ----------------------------------------------------------

    def family(self) -> Iterator[Mobject]:
        """This mobject and all its descendants."""
        yield self
        for child in self.children:
            yield from child.family()

    def signals(self) -> Iterator[Signal]:
        """Every signal of this mobject (not of its children)."""
        yield from self.position.axes
        yield self.rotation
        yield from self.scale.axes
        yield self.alpha
        yield self.present

    @property
    def animate(self) -> AnimationBuilder:
        """Build an animation: ``self.play(square.animate.shift(RIGHT))``."""
        return AnimationBuilder(self)

    # -- authoring reads ----------------------------------------------------

    def get_center(self) -> np.ndarray:
        """Position after everything recorded so far (while composing a scene)."""
        return np.array([authoring.current(axis) for axis in self.position.axes])

    # -- changes ------------------------------------------------------------

    @animatable
    def shift(self, vector: Sequence[float]) -> ChangeList:
        return [(axis, REL, float(component)) for axis, component in zip(self.position.axes, _vector(vector)) if component]

    @animatable
    def move_to(self, point: Sequence[float]) -> ChangeList:
        return [(axis, ABS, float(component)) for axis, component in zip(self.position.axes, _vector(point))]

    @animatable
    def rotate(self, angle: float, axis: Sequence[float] = (0.0, 0.0, 1.0)) -> ChangeList:
        """Rotate by ``angle``; animated, it takes the shortest way (use the
        Rotate animation to sweep full turns)."""
        return [(self.rotation, REL, quaternion_from_axis_angle(axis, angle))]

    @animatable
    def scale_by(self, factor: float | Sequence[float]) -> ChangeList:
        factors = np.broadcast_to(np.asarray(factor, dtype=np.float64), (3,))
        return [(axis, REL, float(f)) for axis, f in zip(self.scale.axes, factors) if f != 1.0]

    @animatable
    def set_scale(self, scale: float | Sequence[float]) -> ChangeList:
        """Set the scale (absolute; ``scale_by`` multiplies instead)."""
        values = np.broadcast_to(np.asarray(scale, dtype=np.float64), (3,))
        return [(axis, ABS, float(v)) for axis, v in zip(self.scale.axes, values)]

    @animatable
    def set_alpha(self, alpha: float) -> ChangeList:
        return [(self.alpha, ABS, alpha)]

    # -- relationships ------------------------------------------------------

    def follow(self, other: Mobject, offset: Sequence[float] = (0.0, 0.0, 0.0)) -> Mobject:
        """Keep this mobject at ``other``'s position plus ``offset`` (a
        derived signal per axis; the replacement for Manim's updaters)."""
        for name, axis, delta in zip("xyz", other.position.axes, _vector(offset)):
            setattr(self.position, name, derive(lambda v, d=float(delta): v + d, axis, name=f"{self.name}.position.{name}"))
        return self

    # -- render state -------------------------------------------------------

    def snapshot(self, alphas: tuple[Signal, ...] = ()) -> RenderObject | None:
        """Render state for the current signal values (None if invisible)."""
        if self._collapsed():
            return None
        alphas = (*alphas, self.alpha)
        children = [snapshot for child in self.children if (snapshot := child.snapshot(alphas)) is not None]
        return RenderGroup(self.name, children=children, transform=self._transform(), uid=self.uid)

    def _collapsed(self) -> bool:
        """Scaled to nothing: invisible, and a zero scale would leave the
        shape's plane without a normal."""
        return any(abs(axis.value) < 1e-9 for axis in self.scale.axes)

    def _transform(self) -> Transform:
        return Transform(position=self.position.value, rotation=self.rotation.value, scale=self.scale.value)


class VMobject(Mobject):
    """A vector shape (Manim's VMobject): a path with fill and stroke.

    ``color`` sets fill and stroke color together, like Manim. The fill is
    transparent unless ``fill_opacity`` is given.
    """

    def __init__(
        self,
        points: Any = (),
        name: str | None = None,
        *,
        color: ParsableManimColor = WHITE,
        fill_color: ParsableManimColor | None = None,
        fill_opacity: float = 0.0,
        stroke_color: ParsableManimColor | None = None,
        stroke_width: float = 4.0,
        stroke_opacity: float = 1.0,
    ) -> None:
        super().__init__(name)
        prefix = self.name
        self.points = RootSignal(np.zeros((0, 2)) if len(points) == 0 else points, PATH, name=f"{prefix}.points")
        self.fill_color = RootSignal(_rgb(fill_color or color), COLOR, name=f"{prefix}.fill_color")
        self.fill_opacity = RootSignal(fill_opacity, name=f"{prefix}.fill_opacity")
        self.stroke_color = RootSignal(_rgb(stroke_color or color), COLOR, name=f"{prefix}.stroke_color")
        self.stroke_width = RootSignal(stroke_width, name=f"{prefix}.stroke_width")
        self.stroke_opacity = RootSignal(stroke_opacity, name=f"{prefix}.stroke_opacity")

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield from (self.points, self.fill_color, self.fill_opacity, self.stroke_color, self.stroke_width, self.stroke_opacity)

    @animatable
    def set_fill(self, color: ParsableManimColor | None = None, opacity: float | None = None) -> ChangeList:
        changes: ChangeList = []
        if color is not None:
            changes.append((self.fill_color, ABS, _rgb(color)))
        if opacity is not None:
            changes.append((self.fill_opacity, ABS, opacity))
        return changes

    @animatable
    def set_stroke(self, color: ParsableManimColor | None = None, width: float | None = None, opacity: float | None = None) -> ChangeList:
        changes: ChangeList = []
        if color is not None:
            changes.append((self.stroke_color, ABS, _rgb(color)))
        if width is not None:
            changes.append((self.stroke_width, ABS, width))
        if opacity is not None:
            changes.append((self.stroke_opacity, ABS, opacity))
        return changes

    @animatable
    def set_color(self, color: ParsableManimColor) -> ChangeList:
        return [(self.fill_color, ABS, _rgb(color)), (self.stroke_color, ABS, _rgb(color))]

    def snapshot(self, alphas: tuple[Signal, ...] = ()) -> RenderObject | None:
        if self._collapsed():
            return None
        alphas = (*alphas, self.alpha)
        alpha = math.prod(signal.value for signal in alphas)
        shape = Shape2D(
            self.name,
            uid=self.uid,
            points=self.points.value,
            fill=Fill(np.array(self.fill_color.value), opacity=min(1.0, max(0.0, self.fill_opacity.value * alpha))),
            stroke=Stroke(
                np.array(self.stroke_color.value),
                opacity=min(1.0, max(0.0, self.stroke_opacity.value * alpha)),
                width=max(0.0, self.stroke_width.value),
            ),
            transform=self._transform(),
            versions={
                "points": version_token(self.points),
                "fill": version_token(self.fill_color, self.fill_opacity, *alphas),
                "stroke": version_token(self.stroke_color, self.stroke_width, self.stroke_opacity, *alphas),
            },
        )
        if not self.children:
            return shape
        children = [snapshot for child in self.children if (snapshot := child.snapshot(alphas)) is not None]
        return RenderGroup(f"{self.name}-family", children=[shape, *children], uid=new_uid())


class Group(Mobject):
    """A mobject that only holds others; its transform and alpha apply to all."""

    def __init__(self, *children: Mobject, name: str | None = None) -> None:
        super().__init__(name)
        self.children = list(children)

    def add(self, *children: Mobject) -> Group:
        self.children.extend(children)
        return self


class ClippingGroup(Mobject):
    """Shows ``content`` only inside ``mask`` (the renderer's soft alpha
    mask: the mask's fill and stroke opacity, as drawn, multiply the
    content). The mask itself is never drawn. Transform and alpha apply to
    both, so scaling the group scales the mask with its content."""

    def __init__(self, mask: Mobject, content: Mobject, name: str | None = None, one_sided: bool = False) -> None:
        super().__init__(name)
        self.mask = mask
        self.content = content
        self.one_sided = one_sided
        self.children = [mask, content]

    def snapshot(self, alphas: tuple[Signal, ...] = ()) -> RenderObject | None:
        if self._collapsed():
            return None
        alphas = (*alphas, self.alpha)
        mask, content = self.mask.snapshot(alphas), self.content.snapshot(alphas)
        if mask is None or content is None:
            return None  # nothing to show through
        return RenderClippingGroup(
            self.name, mask=mask, content=content, one_sided=self.one_sided, transform=self._transform(), uid=self.uid
        )


# --------------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------------


class Circle(VMobject):
    def __init__(self, radius: float = 1.0, name: str | None = None, **style: Any) -> None:
        super().__init__(arc_points(radius), name, **style)


class Dot(VMobject):
    def __init__(self, radius: float = 0.08, name: str | None = None, **style: Any) -> None:
        style.setdefault("fill_opacity", 1.0)
        style.setdefault("stroke_width", 0.0)
        super().__init__(arc_points(radius), name, **style)


class Rectangle(VMobject):
    def __init__(self, width: float = 4.0, height: float = 2.0, name: str | None = None, **style: Any) -> None:
        super().__init__(rectangle("rectangle", width, height).points, name, **style)


class Square(VMobject):
    def __init__(self, side_length: float = 2.0, name: str | None = None, **style: Any) -> None:
        super().__init__(rectangle("square", side_length, side_length).points, name, **style)


class RegularPolygon(VMobject):
    def __init__(self, n: int = 6, radius: float = 1.0, name: str | None = None, **style: Any) -> None:
        super().__init__(regular_polygon("polygon", n, radius).points, name, **style)


class Polyline(VMobject):
    """Straight segments through ``corners`` (closed into a polygon if asked)."""

    def __init__(self, corners: Sequence[Sequence[float]], closed: bool = False, name: str | None = None, **style: Any) -> None:
        super().__init__(Shape2D.from_corners("polyline", corners, closed=closed).points, name, **style)


class Triangle(RegularPolygon):
    def __init__(self, radius: float = 1.0, name: str | None = None, **style: Any) -> None:
        super().__init__(3, radius, name, **style)


def _vector(value: Sequence[float]) -> tuple[float, float, float]:
    array = np.zeros(3)
    given = np.asarray(value, dtype=np.float64).reshape(-1)
    array[: len(given)] = given
    return (float(array[0]), float(array[1]), float(array[2]))


__all__ = [
    "Circle",
    "ClippingGroup",
    "Dot",
    "Group",
    "Mobject",
    "Polyline",
    "Rectangle",
    "RegularPolygon",
    "Square",
    "Triangle",
    "VMobject",
    "animatable",
]
