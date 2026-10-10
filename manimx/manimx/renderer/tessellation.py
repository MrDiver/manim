"""CPU tessellation of :class:`RShape2D` fills into triangles with exact curves.

The work is done by the core (core/src/tessellation.rs), for the core's
packer and for the Python renderer alike, so both produce the same
triangles. The approach is the one ManimGL uses (Loop & Blinn,
"Resolution Independent Curve Rendering using Programmable Graphics
Hardware", 2005):

1. Every cubic Bezier is split into quadratic Beziers with a bounded error.
2. Every subpath becomes a polygon *ring*. A quadratic that bulges *into* the
   filled region (concave) contributes its control point to the ring; one that
   bulges *out of* it (convex) contributes only its chord.
3. Rings are grouped into outer rings and holes (even-odd rule) and
   triangulated with earcut.
4. Each quadratic additionally emits one *curve triangle* (start, control,
   end). Its fragment shader evaluates the implicit curve ``u^2 - v`` and
   discards the half that lies outside the shape, which makes the edge exact
   at any zoom level:

       convex:  polygon stops at the chord, curve triangle adds the bulge
       concave: polygon goes around the control point, curve triangle adds
                the part between the curve and the control point

Everything is in the shape's local 2D coordinates; transforms are applied on
the GPU, so moving a shape never re-tessellates it.

The mesh holds geometry only. Instead of a color, every vertex records
where it comes from: the index of its source cubic and the parameter ``t``
on it. The GPU computes the vertex color from the shape's per-point fill
colors, the same cubic Bezier of the curve's four point colors at ``t``.
So a triangulation depends on the points alone, and fades or color changes
never re-tessellate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import _core

SIDE_SOLID = 0.0
"""Triangle is entirely inside the shape."""
SIDE_CONTROL = 1.0
"""Keep the control-point side of the curve (concave curve)."""
SIDE_CHORD = -1.0
"""Keep the chord side of the curve (convex curve)."""

FILL_VERTEX_DTYPE = np.dtype(
    [("position", "<f4", 2), ("uv", "<f4", 2), ("side", "<f4"), ("curve", "<u4"), ("t", "<f4"), ("_pad", "<f4")]
)
"""The records the core returns (and the GPU reads, see vector_batch.py)."""


@dataclass(frozen=True, slots=True, eq=False)
class FillMesh:
    """Non-indexed triangle list, three consecutive vertices per triangle."""

    positions: np.ndarray  # (M, 2) local coordinates
    uv: np.ndarray  # (M, 2) implicit-curve coordinates
    side: np.ndarray  # (M,) one of the SIDE_* constants
    curve: np.ndarray  # (M,) index of the source cubic (colors come from its points)
    t: np.ndarray  # (M,) parameter on the source cubic

    def __len__(self) -> int:
        return len(self.positions)

    @staticmethod
    def empty() -> FillMesh:
        return FillMesh(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0), np.zeros(0, np.uint32), np.zeros(0))


def tessellate_fill(points: np.ndarray) -> FillMesh:
    """Triangulate an :class:`RBezierPath`-layout point array (see :class:`RShape2D`).

    Open subpaths are closed with a straight line, like Cairo does when
    filling.
    """
    records = np.frombuffer(_core.tessellate_fill(np.ascontiguousarray(points[:, :2], dtype=np.float64)), FILL_VERTEX_DTYPE)
    if len(records) == 0:
        return FillMesh.empty()
    return FillMesh(
        positions=records["position"].astype(np.float64),
        uv=records["uv"].astype(np.float64),
        side=records["side"].astype(np.float64),
        curve=records["curve"].astype(np.uint32),
        t=records["t"].astype(np.float64),
    )


__all__ = ["FILL_VERTEX_DTYPE", "SIDE_CHORD", "SIDE_CONTROL", "SIDE_SOLID", "FillMesh", "tessellate_fill"]
