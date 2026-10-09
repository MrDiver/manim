"""Shape3D: a lit triangle mesh.

A :class:`Shape3D` snapshots to an :class:`~manimx.robjects.RShape3D`.
Every field of the render counterpart is a signal; the material's fields
are signals of their own, so each can be animated:

=============================  ========  =================================
signal                         kind      RShape3D field
=============================  ========  =================================
``mesh``                       DISCRETE  ``mesh`` (a MeshData)
``base_color``                 COLOR     ``material.base_color`` (sRGB)
``opacity``                    SCALAR    ``material.opacity`` (times every
                                         ``alpha``)
``metallic``                   SCALAR    ``material.metallic``
``roughness``                  SCALAR    ``material.roughness``
``emissive``                   COLOR     ``material.emissive`` (sRGB)
``emissive_strength``          SCALAR    ``material.emissive_strength``
``normal_scale``               SCALAR    ``material.normal_scale``
``occlusion_strength``         SCALAR    ``material.occlusion_strength``
``base_color_texture``         DISCRETE  ``material.base_color_texture``
``metallic_roughness_texture`` DISCRETE  ``material.metallic_roughness_texture``
``normal_texture``             DISCRETE  ``material.normal_texture``
``occlusion_texture``          DISCRETE  ``material.occlusion_texture``
``emissive_texture``           DISCRETE  ``material.emissive_texture``
``cast_shadows``               DISCRETE  ``cast_shadows``
=============================  ========  =================================

The mesh signal's version (and with it the ``mesh`` version token of
every snapshot) changes only when another MeshData is set: DISCRETE
compares MeshData by identity. So the renderer's per-mesh caches keep
hitting while the shape moves, turns, scales or changes its material.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from ..robjects import (
    MeshData,
    PbrMaterial,
    RNode,
    RShape3D,
    Texture,
)
from ..utils.color import ParsableManimColor
from .kinds import COLOR, DISCRETE, SCALAR, SignalKind
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

    ``material`` gives the initial value of every material signal;
    ``color``, ``metallic``, ``roughness``, ``opacity``, ``emissive`` and
    ``emissive_strength`` override single ones.
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

        def signal(field: str, value: object, kind: SignalKind) -> RootSignal:
            return RootSignal(value, kind, name=f"{prefix}.{field}")

        self.mesh = signal("mesh", mesh if mesh is not None else _no_mesh(), DISCRETE)
        self.base_color = signal("base_color", _rgb(base.base_color if color is None else color), COLOR)
        self.opacity = signal("opacity", base.opacity if opacity is None else opacity, SCALAR)
        self.metallic = signal("metallic", base.metallic if metallic is None else metallic, SCALAR)
        self.roughness = signal("roughness", base.roughness if roughness is None else roughness, SCALAR)
        self.emissive = signal("emissive", _rgb(base.emissive if emissive is None else emissive), COLOR)
        self.emissive_strength = signal(
            "emissive_strength", base.emissive_strength if emissive_strength is None else emissive_strength, SCALAR
        )
        self.normal_scale = signal("normal_scale", base.normal_scale, SCALAR)
        self.occlusion_strength = signal("occlusion_strength", base.occlusion_strength, SCALAR)
        self.base_color_texture = signal("base_color_texture", base.base_color_texture, DISCRETE)
        self.metallic_roughness_texture = signal("metallic_roughness_texture", base.metallic_roughness_texture, DISCRETE)
        self.normal_texture = signal("normal_texture", base.normal_texture, DISCRETE)
        self.occlusion_texture = signal("occlusion_texture", base.occlusion_texture, DISCRETE)
        self.emissive_texture = signal("emissive_texture", base.emissive_texture, DISCRETE)
        self.cast_shadows = signal("cast_shadows", cast_shadows, DISCRETE)

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield self.mesh
        yield from self._material_signals()
        yield self.cast_shadows

    def _material_signals(self) -> tuple[RootSignal, ...]:
        """Every signal the snapshot's material is built from."""
        return (
            self.base_color, self.opacity, self.metallic, self.roughness, self.emissive,
            self.emissive_strength, self.normal_scale, self.occlusion_strength, *self._texture_signals(),
        )  # fmt: skip

    def _texture_signals(self) -> tuple[RootSignal, ...]:
        return (
            self.base_color_texture, self.metallic_roughness_texture, self.normal_texture,
            self.occlusion_texture, self.emissive_texture,
        )  # fmt: skip

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
        normal_scale: float | None = None,
        occlusion_strength: float | None = None,
    ) -> ChangeList:
        changes: ChangeList = []
        if color is not None:
            changes.append((self.base_color, ABS, _rgb(color)))
        for signal, value in (
            (self.opacity, opacity), (self.metallic, metallic), (self.roughness, roughness),
            (self.normal_scale, normal_scale), (self.occlusion_strength, occlusion_strength),
        ):  # fmt: skip
            if value is not None:
                changes.append((signal, ABS, value))
        return changes

    @animatable
    def set_textures(self, **textures: Texture | None) -> ChangeList:
        """Switch textures by field name (``base_color``, ``metallic_roughness``,
        ``normal``, ``occlusion``, ``emissive``); None removes one."""
        changes: ChangeList = []
        for field, texture in textures.items():
            signal = getattr(self, f"{field}_texture", None)
            if signal not in self._texture_signals():
                raise TypeError(f"Shape3D has no {field}_texture; use one of base_color, metallic_roughness, normal, occlusion, emissive")
            if texture is not None and not isinstance(texture, Texture):
                raise TypeError(f"a texture must be a Texture or None, got {type(texture).__name__}")
            changes.append((signal, ABS, texture))
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

    def _build_snapshot(self) -> RNode | None:
        if self._hidden():
            return None
        material = PbrMaterial(
            base_color=np.array(self.base_color.value),
            opacity=_unit(self.opacity.value),
            metallic=_unit(self.metallic.value),
            roughness=_unit(self.roughness.value),
            emissive=np.array(self.emissive.value),
            emissive_strength=max(0.0, self.emissive_strength.value),
            normal_scale=self.normal_scale.value,
            occlusion_strength=_unit(self.occlusion_strength.value),
            base_color_texture=self.base_color_texture.value,
            metallic_roughness_texture=self.metallic_roughness_texture.value,
            normal_texture=self.normal_texture.value,
            occlusion_texture=self.occlusion_texture.value,
            emissive_texture=self.emissive_texture.value,
        )
        return RShape3D(
            uid=self.uid,
            mesh=self.mesh.value,
            material=material,
            cast_shadows=bool(self.cast_shadows.value),
            transform=self._transform(),
            alpha=self.alpha.value,
            versions={
                "mesh": version_token(self.mesh),
                "material": version_token(*self._material_signals()),
            },
        )


def _no_mesh() -> MeshData:
    """An empty mesh: nothing to draw until one is set."""
    return MeshData(np.zeros((0, 3)), np.zeros(0, dtype=np.uint32))


def _unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


__all__ = ["Shape3D"]
