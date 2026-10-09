"""Lights: nodes that light Shape3D meshes; they snapshot to the RLight classes.

Every light has ``color`` (COLOR), ``intensity`` and ``size`` (SCALAR) and
``cast_shadows`` (DISCRETE). ``alpha`` multiplies the intensity, so
``FadeIn(light)`` and a fading group fade the light. Lights shine along
their local -z axis.

Each kind takes only the transform parts that mean something for it:

* :class:`DirectionalLight`: a rotation (its direction), no position;
* :class:`PointLight`: a position, no rotation;
* :class:`SpotLight`: both.

No light can be scaled.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence

import numpy as np

from ..objects import RDirectionalLight, RLight, RNode, RPointLight, RSpotLight
from ..objects.transform import rotation_towards
from ..utils.color import WHITE, ParsableManimColor
from . import authoring
from .kinds import COLOR, DISCRETE, SCALAR
from .node import (
    ABS,
    ChangeList,
    Node,
    PositionMixin,
    RotationMixin,
    _rgb,
    _vector,
    animatable,
)
from .signals import RootSignal, Signal


class Light(Node):
    """The signals every light has (see the module docs)."""

    def __init__(
        self,
        name: str | None = None,
        *,
        color: ParsableManimColor = WHITE,
        intensity: float = 1.0,
        size: float = 0.1,
        cast_shadows: bool = True,
        tags: Iterable[str] = (),
    ) -> None:
        super().__init__(name, tags=tags)
        prefix = self.name
        self.color = RootSignal(_rgb(color), COLOR, name=f"{prefix}.color")
        self.intensity = RootSignal(intensity, SCALAR, name=f"{prefix}.intensity")
        self.size = RootSignal(size, SCALAR, name=f"{prefix}.size")
        self.cast_shadows = RootSignal(cast_shadows, DISCRETE, name=f"{prefix}.cast_shadows")

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield from (self.color, self.intensity, self.size, self.cast_shadows)

    @animatable
    def set_color(self, color: ParsableManimColor) -> ChangeList:
        return [(self.color, ABS, _rgb(color))]

    @animatable
    def set_intensity(self, intensity: float) -> ChangeList:
        return [(self.intensity, ABS, intensity)]

    @animatable
    def set_size(self, size: float) -> ChangeList:
        return [(self.size, ABS, size)]

    @animatable
    def set_cast_shadows(self, cast_shadows: bool) -> ChangeList:
        return [(self.cast_shadows, ABS, cast_shadows)]

    def snapshot(self, alphas: tuple[Signal, ...] = ()) -> RNode | None:
        if self._hidden():
            return None
        alpha = math.prod(signal.value for signal in (*alphas, self.alpha))
        return self._render_class()(
            uid=self.uid,
            color=np.array(self.color.value),
            intensity=max(0.0, self.intensity.value * alpha),
            size=max(0.0, self.size.value),
            cast_shadows=bool(self.cast_shadows.value),
            transform=self._transform(),
            **self._render_fields(),
        )

    def _render_class(self) -> type[RLight]:
        raise NotImplementedError

    def _render_fields(self) -> dict[str, object]:
        """Fields of the render class beyond those of every light."""
        return {}


class DirectionalLight(RotationMixin, Light):
    """Parallel light (like the sun): only its direction matters."""

    def __init__(
        self,
        name: str | None = None,
        *,
        direction: Sequence[float] = (0.0, 0.0, -1.0),
        intensity: float = 3.0,
        size: float = 0.03,
        **light: object,
    ) -> None:
        super().__init__(name, intensity=intensity, size=size, **light)  # type: ignore[arg-type]
        self.rotation.set(rotation_towards(_vector(direction)))

    @animatable
    def point_along(self, direction: Sequence[float]) -> ChangeList:
        """Shine in world ``direction`` (animated, it turns the short way)."""
        return [(self.rotation, ABS, rotation_towards(_vector(direction)))]

    def _render_class(self) -> type[RLight]:
        return RDirectionalLight


class PointLight(PositionMixin, Light):
    """Light emitted in all directions from its position; ``range`` (0 for
    none) fades it to zero at that distance."""

    def __init__(
        self,
        name: str | None = None,
        *,
        position: Sequence[float] = (0.0, 0.0, 0.0),
        intensity: float = 20.0,
        range: float = 0.0,
        **light: object,
    ) -> None:
        super().__init__(name, intensity=intensity, **light)  # type: ignore[arg-type]
        self.position.set(_vector(position))
        self.range = RootSignal(range, SCALAR, name=f"{self.name}.range")

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield self.range

    @animatable
    def set_range(self, range: float) -> ChangeList:
        return [(self.range, ABS, range)]

    def _render_class(self) -> type[RLight]:
        return RPointLight

    def _render_fields(self) -> dict[str, object]:
        return {"range": _range(self.range.value)}


class SpotLight(PositionMixin, RotationMixin, Light):
    """A point light restricted to a cone around its local -z axis: full
    intensity inside ``inner_cone_angle``, fading to zero at
    ``outer_cone_angle`` (radians from the axis). While animated, the
    angles are kept valid (0 <= inner < outer <= pi / 2)."""

    def __init__(
        self,
        name: str | None = None,
        *,
        position: Sequence[float] = (0.0, 0.0, 0.0),
        target: Sequence[float] | None = None,
        intensity: float = 40.0,
        range: float = 0.0,
        inner_cone_angle: float = 0.0,
        outer_cone_angle: float = math.pi / 4,
        **light: object,
    ) -> None:
        super().__init__(name, intensity=intensity, **light)  # type: ignore[arg-type]
        self.position.set(_vector(position))
        if target is not None:
            self.rotation.set(rotation_towards(tuple(np.subtract(_vector(target), _vector(position)))))
        prefix = self.name
        self.range = RootSignal(range, SCALAR, name=f"{prefix}.range")
        self.inner_cone_angle = RootSignal(inner_cone_angle, SCALAR, name=f"{prefix}.inner_cone_angle")
        self.outer_cone_angle = RootSignal(outer_cone_angle, SCALAR, name=f"{prefix}.outer_cone_angle")

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield from (self.range, self.inner_cone_angle, self.outer_cone_angle)

    @animatable
    def point_along(self, direction: Sequence[float]) -> ChangeList:
        """Shine in world ``direction``."""
        return [(self.rotation, ABS, rotation_towards(_vector(direction)))]

    @animatable
    def point_at(self, target: Sequence[float]) -> ChangeList:
        """Shine at ``target`` from where the light is at this point of the
        scene."""
        here = np.array([authoring.current(axis) for axis in self.position.axes])
        return [(self.rotation, ABS, rotation_towards(tuple(np.subtract(_vector(target), here))))]

    @animatable
    def set_cone(self, inner: float | None = None, outer: float | None = None) -> ChangeList:
        changes: ChangeList = []
        if inner is not None:
            changes.append((self.inner_cone_angle, ABS, inner))
        if outer is not None:
            changes.append((self.outer_cone_angle, ABS, outer))
        return changes

    @animatable
    def set_range(self, range: float) -> ChangeList:
        return [(self.range, ABS, range)]

    def _render_class(self) -> type[RLight]:
        return RSpotLight

    def _render_fields(self) -> dict[str, object]:
        outer = min(math.pi / 2, max(1e-6, self.outer_cone_angle.value))
        inner = min(max(0.0, self.inner_cone_angle.value), outer * (1.0 - 1e-6))
        return {"range": _range(self.range.value), "inner_cone_angle": inner, "outer_cone_angle": outer}


def _range(value: float) -> float | None:
    return value if value > 0.0 else None


__all__ = ["DirectionalLight", "Light", "PointLight", "SpotLight"]
