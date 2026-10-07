from __future__ import annotations

from html import escape

import numpy as np

from ..objects import (
    CapStyle,
    ClippingGroup,
    Fill,
    Group,
    JointType,
    RenderObject,
    Shape2D,
    Shape3D,
    Stroke,
)
from ..objects.shape2d import is_closed
from ..objects.transform import apply_matrix

_JOINTS = {JointType.ROUND: "round", JointType.BEVEL: "bevel", JointType.MITER: "miter"}
_CAPS = {CapStyle.ROUND: "round", CapStyle.BUTT: "butt", CapStyle.SQUARE: "square"}


def _number(value: float) -> str:
    # Round away float noise from rotations and normalize -0.0.
    return f"{round(float(value), 6) + 0.0:g}"


def _hex(color: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{round(c * 255):02x}" for c in color)


class SVGRenderer:
    """Reference renderer: orthographic projection of world space onto XY."""

    def render(self, obj: RenderObject) -> str:
        body = self._render_object(obj, np.eye(4))
        return f'<svg height="100" viewBox="-50 -50 100 100" width="100">{body}</svg>'

    def _render_object(
        self, obj: RenderObject, parent: np.ndarray, in_mask: bool = False
    ) -> str:
        if not obj.visible:
            return ""
        world = parent @ obj.transform.matrix()
        if isinstance(obj, Shape2D):
            return self._render_shape_2d(obj, world)
        if isinstance(obj, Shape3D):
            return self._render_shape_3d(obj, world)
        if isinstance(obj, Group):
            return "".join(
                self._render_object(child, world, in_mask) for child in obj.children
            )
        if isinstance(obj, ClippingGroup):
            if in_mask:
                raise ValueError("nested ClippingGroup inside a mask is not supported")
            return self._render_clipping_group(obj, world)
        raise TypeError(f"unsupported render object: {type(obj).__name__}")

    def _render_clipping_group(self, obj: ClippingGroup, world: np.ndarray) -> str:
        if obj.content is None:
            return ""
        content = self._render_object(obj.content, world)
        if obj.mask is None:
            return content
        clip_id = escape(f"clip-{obj.id}")
        mask = self._render_object(obj.mask, world, in_mask=True)
        return (
            f'<clipPath id="{clip_id}">{mask}</clipPath>'
            f'<g clip-path="url(#{clip_id})" data-id="{escape(obj.id)}">{content}</g>'
        )

    def _render_shape_2d(self, obj: Shape2D, world: np.ndarray) -> str:
        commands = []
        for subpath in obj.subpaths():
            points = apply_matrix(world, np.column_stack([subpath, np.zeros(len(subpath))]))
            xy = [f"{_number(x)},{_number(y)}" for x, y, _ in points]
            commands.append(f"M {xy[0]}")
            for index in range(0, len(xy), 4):
                commands.append(f"C {xy[index + 1]} {xy[index + 2]} {xy[index + 3]}")
            if is_closed(subpath):
                commands.append("Z")
        d = " ".join(commands)
        data_id = escape(obj.id)
        background = ""
        if obj.background_stroke is not None and obj.background_stroke.width > 0:
            background = (
                f'<path data-id="{data_id}" d="{d}" fill="none"'
                f"{self._stroke_attributes(obj.background_stroke)} />"
            )
        return (
            f'{background}<path data-id="{data_id}" d="{d}"'
            f"{self._fill_attributes(obj.fill)}"
            f"{self._stroke_attributes(obj.stroke)} />"
        )

    def _fill_attributes(self, fill: Fill) -> str:
        if fill.opacity == 0.0:
            return ' fill="none"'
        return f' fill="{_hex(fill.color)}" fill-opacity="{_number(fill.opacity)}"'

    def _stroke_attributes(self, stroke: Stroke) -> str:
        if stroke.width == 0.0 or stroke.opacity == 0.0:
            return ""
        attributes = (
            f' stroke="{_hex(stroke.color)}"'
            f' stroke-opacity="{_number(stroke.opacity)}"'
            f' stroke-width="{_number(stroke.width)}"'
        )
        if stroke.joint in _JOINTS:
            attributes += f' stroke-linejoin="{_JOINTS[stroke.joint]}"'
        if stroke.cap in _CAPS:
            attributes += f' stroke-linecap="{_CAPS[stroke.cap]}"'
        return attributes

    def _render_shape_3d(self, obj: Shape3D, world: np.ndarray) -> str:
        vertices = apply_matrix(world, np.asarray(obj.vertices, dtype=np.float64).reshape(-1, 3))
        points = " ".join(f"{_number(x)},{_number(y)}" for x, y, _ in vertices)
        return f'<polygon data-id="{escape(obj.id)}" points="{points}" />'
