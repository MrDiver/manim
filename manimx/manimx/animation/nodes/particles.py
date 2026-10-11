"""Particles: many copies of one mesh in one node.

A firework of a thousand sparks as a thousand Shape3D nodes costs a
thousand snapshots, a thousand tracks per signal and a thousand draws
in every frame. :class:`Particles` keeps them in one node instead: the
mesh and the material are shared (the signals of :class:`Shape3D`), and
each particle is a row of four array signals:

=============  =======  ===========================================
signal         shape    meaning
=============  =======  ===========================================
``positions``  (N, 3)   where each particle is (in the node's space)
``rotations``  (N, 4)   its orientation, a quaternion ``w, x, y, z``
``scales``     (N, 3)   its size per axis; 0 on any axis hides it
``colors``     (N, 4)   ``r, g, b`` multiply the material's colors,
                        ``glow`` its emissive strength
=============  =======  ===========================================

The arrays interpolate element-wise (kind ``ARRAY``), so a Simulation
records and replays them as four signals, whatever N is. The node
snapshots to one :class:`~manimx.robjects.RParticles`, which the
renderer draws with one instanced draw.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from ...robjects import MeshData
from ..signals.kinds import ARRAY
from ..signals.signal import RootSignal, Signal
from .node import ABS, ChangeList, animatable
from .shape3d import Shape3D


class Particles(Shape3D):
    """``count`` copies of ``mesh``: at the node's origin, unrotated, at
    full size, untinted, until set. The keyword arguments are those of
    :class:`Shape3D` and apply to all particles."""

    render_kind = "particles"

    def __init__(self, count: int, mesh: MeshData | None = None, name: str | None = None, **style: Any) -> None:
        super().__init__(mesh, name, **style)
        self.count = count
        identity = np.zeros((count, 4))
        identity[:, 0] = 1.0

        def signal(field: str, value: np.ndarray) -> RootSignal:
            return RootSignal(ARRAY.validate(value), ARRAY, name=f"{self.name}.{field}")

        self.positions = signal("positions", np.zeros((count, 3)))
        self.rotations = signal("rotations", identity)
        self.scales = signal("scales", np.ones((count, 3)))
        self.colors = signal("colors", np.ones((count, 4)))

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()
        yield from (self.positions, self.rotations, self.scales, self.colors)

    def _rows(self, values: Any, columns: int) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self.count, columns):
            raise ValueError(f"{self.name} has {self.count} particles: expected shape ({self.count}, {columns}), got {array.shape}")
        return array

    @animatable
    def set_positions(self, positions: Any) -> ChangeList:
        return [(self.positions, ABS, self._rows(positions, 3))]

    @animatable
    def set_rotations(self, rotations: Any) -> ChangeList:
        return [(self.rotations, ABS, self._rows(rotations, 4))]

    @animatable
    def set_scales(self, scales: Any) -> ChangeList:
        return [(self.scales, ABS, self._rows(scales, 3))]

    @animatable
    def set_colors(self, colors: Any) -> ChangeList:
        return [(self.colors, ABS, self._rows(colors, 4))]

    def render_fields(self) -> Iterator[tuple[str, Signal]]:
        yield from super().render_fields()
        yield from (("positions", self.positions), ("rotations", self.rotations), ("scales", self.scales), ("colors", self.colors))


__all__ = ["Particles"]
