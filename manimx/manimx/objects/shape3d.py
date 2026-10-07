from __future__ import annotations

from dataclasses import dataclass, field

from .render_object import RenderObject


@dataclass(slots=True, eq=False)
class Shape3D(RenderObject):
    vertices: list[tuple[float, float, float]] = field(default_factory=list)
