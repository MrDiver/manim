"""Shape3D: a lit triangle mesh, and the solids built on it.

A :class:`Shape3D` snapshots to an :class:`~manimx.robjects.RShape3D`. Its
mesh is one signal holding a :class:`~manimx.robjects.MeshData`, and the
factors of its material are signals of their own:

* ``mesh``: DISCRETE. Its version (and with it the ``mesh`` version token
  of every snapshot) changes only when another MeshData is set, so the
  renderer's per-mesh caches keep hitting while the shape moves, turns,
  scales or changes color.
* ``base_color``, ``emissive``: COLOR, sRGB like all colors.
* ``opacity``, ``metallic``, ``roughness``, ``emissive_strength``: SCALAR.
* ``cast_shadows``: DISCRETE.

Textures, ``normal_scale`` and ``occlusion_strength`` are not animated:
they come from the ``material`` given to the constructor, which also
supplies the initial factors. ``alpha`` multiplies ``opacity``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
from typing import Any

import numpy as np

from ..robjects import (
    MeshData,
    PbrMaterial,
    RNode,
    RShape3D,
    box_mesh,
    sphere_mesh,
    torus_mesh,
)
from ..robjects.mesh import surface_mesh
from ..utils.color import ParsableManimColor
from .kinds import COLOR, DISCRETE, SCALAR
from .node import (
    ABS,
    ChangeList,
    Node,
    PositionMixin,
    RotationMixin,
    ScalableMixin,
    _rgb,
    animatable,
)
from .signals import RootSignal, Signal
from .snapshot import version_token


class Shape3D(PositionMixin, RotationMixin, ScalableMixin, Node):
    """A mesh with a metallic-roughness material (see the module docs).

    ``color``, ``metallic``, ``roughness``, ``opacity``, ``emissive`` and
    ``emissive_strength`` override the factors of ``material``.
    """

    def __init__(
        self,
        mesh: MeshData | None = None,
        name: str | None = None,
        *,
        material: PbrMaterial | None = None,
        color: ParsableManimColor | None = None,
        metallic: float | None = None,
        roughness: float | None = None,
        opacity: float | None = None,
        emissive: ParsableManimColor | None = None,
        emissive_strength: float | None = None,
        cast_shadows: bool = True,
        tags: Iterable[str] = (),
    ) -> None:
        super().__init__(name, tags=tags)
        base = material or PbrMaterial()
        prefix = self.name
        self.material = base
        """The constant part of the material: textures, normal_scale and
        occlusion_strength (its factors only give the initial values)."""
        self.mesh = RootSignal(mesh if mesh is not None else box_mesh(), DISCRETE, name=f"{prefix}.mesh")
        self.base_color = RootSignal(_rgb(base.base_color if color is None else color), COLOR, name=f"{prefix}.base_color")
        self.opacity = RootSignal(base.opacity if opacity is None else opacity, SCALAR, name=f"{prefix}.opacity")
        self.metallic = RootSignal(base.metallic if metallic is None else metallic, SCALAR, name=f"{prefix}.metallic")
        self.roughness = RootSignal(base.roughness if roughness is None else roughness, SCALAR, name=f"{prefix}.roughness")
        self.emissive = RootSignal(_rgb(base.emissive if emissive is None else emissive), COLOR, name=f"{prefix}.emissive")
        self.emissive_strength = RootSignal(
            base.emissive_strength if emissive_strength is None else emissive_strength, SCALAR, name=f"{prefix}.emissive_strength"
        )
        self.cast_shadows = RootSignal(cast_shadows, DISCRETE, name=f"{prefix}.cast_shadows")

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield from (
            self.mesh, self.base_color, self.opacity, self.metallic, self.roughness,
            self.emissive, self.emissive_strength, self.cast_shadows,
        )  # fmt: skip

    def _material_signals(self) -> tuple[RootSignal, ...]:
        return (self.base_color, self.opacity, self.metallic, self.roughness, self.emissive, self.emissive_strength)

    @animatable
    def set_mesh(self, mesh: MeshData) -> ChangeList:
        """Switch to another mesh (at the end of an animation)."""
        if not isinstance(mesh, MeshData):
            raise TypeError(f"a mesh must be a MeshData, got {type(mesh).__name__}")
        return [(self.mesh, ABS, mesh)]

    @animatable
    def set_material(
        self,
        color: ParsableManimColor | None = None,
        opacity: float | None = None,
        metallic: float | None = None,
        roughness: float | None = None,
    ) -> ChangeList:
        changes: ChangeList = []
        if color is not None:
            changes.append((self.base_color, ABS, _rgb(color)))
        for signal, value in ((self.opacity, opacity), (self.metallic, metallic), (self.roughness, roughness)):
            if value is not None:
                changes.append((signal, ABS, value))
        return changes

    @animatable
    def set_color(self, color: ParsableManimColor) -> ChangeList:
        return [(self.base_color, ABS, _rgb(color))]

    @animatable
    def set_emissive(self, color: ParsableManimColor | None = None, strength: float | None = None) -> ChangeList:
        changes: ChangeList = []
        if color is not None:
            changes.append((self.emissive, ABS, _rgb(color)))
        if strength is not None:
            changes.append((self.emissive_strength, ABS, strength))
        return changes

    @animatable
    def set_cast_shadows(self, cast_shadows: bool) -> ChangeList:
        return [(self.cast_shadows, ABS, cast_shadows)]

    def snapshot(self, alphas: tuple[Signal, ...] = ()) -> RNode | None:
        if self._hidden():
            return None
        alphas = (*alphas, self.alpha)
        alpha = math.prod(signal.value for signal in alphas)
        material = replace(
            self.material,
            base_color=np.array(self.base_color.value),
            opacity=_unit(self.opacity.value * alpha),
            metallic=_unit(self.metallic.value),
            roughness=_unit(self.roughness.value),
            emissive=np.array(self.emissive.value),
            emissive_strength=max(0.0, self.emissive_strength.value),
        )
        return RShape3D(
            uid=self.uid,
            mesh=self.mesh.value,
            material=material,
            cast_shadows=bool(self.cast_shadows.value),
            transform=self._transform(),
            versions={
                "mesh": version_token(self.mesh),
                "material": version_token(*self._material_signals(), *alphas),
            },
        )


class Box(Shape3D):
    """An axis-aligned box centered at the origin."""

    def __init__(self, width: float = 2.0, height: float = 2.0, depth: float = 2.0, name: str | None = None, **style: Any) -> None:
        super().__init__(box_mesh(width, height, depth), name, **style)


class Sphere(Shape3D):
    """A UV sphere around the z axis."""

    def __init__(self, radius: float = 1.0, segments: int = 48, rings: int = 24, name: str | None = None, **style: Any) -> None:
        super().__init__(sphere_mesh(radius, segments, rings), name, **style)


class Torus(Shape3D):
    """A torus around the z axis."""

    def __init__(
        self,
        major_radius: float = 1.0,
        minor_radius: float = 0.35,
        segments: int = 64,
        sides: int = 32,
        name: str | None = None,
        **style: Any,
    ) -> None:
        super().__init__(torus_mesh(major_radius, minor_radius, segments, sides), name, **style)


class ParametricSurface(Shape3D):
    """The surface through ``function(u, v)`` (a 3D point), sampled on a
    grid of ``resolution`` cells (see robjects.mesh.surface_mesh). Surfaces
    of the same resolution have corresponding vertices."""

    def __init__(
        self,
        function: Callable[[float, float], Any],
        u_range: tuple[float, float] = (0.0, 1.0),
        v_range: tuple[float, float] = (0.0, 1.0),
        resolution: tuple[int, int] = (32, 32),
        name: str | None = None,
        **style: Any,
    ) -> None:
        super().__init__(surface_mesh(function, u_range, v_range, resolution), name, **style)


def _unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


__all__ = ["Box", "ParametricSurface", "Shape3D", "Sphere", "Torus"]
