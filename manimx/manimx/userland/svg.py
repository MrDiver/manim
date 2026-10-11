"""Reading SVG files into Shape2D nodes.

:func:`read_svg` turns an SVG document into :class:`SVGPath` values: the
points of every drawn element in the RBezierPath layout (four points per
cubic curve), with the fill and stroke the document gives it. :class:`SVG`
builds a Group with one Shape2D per path from them.

Supported: ``path`` (all commands, arcs included), ``rect``, ``circle``,
``ellipse``, ``line``, ``polyline``, ``polygon``, ``g``, ``use`` (also of
``symbol``), ``transform`` attributes, presentation attributes and
``style`` declarations, inherited down the tree. Not supported: text
(convert it to paths first), images, clip paths, masks and filters. A
gradient paint becomes the color of its first stop.

Units: a scene unit is one inch, so an SVG's physical size (its ``width``
and ``height`` in pt, mm, px, ...) is kept unless a size is asked for.
"""

from __future__ import annotations

import itertools
import math
import re
import xml.etree.ElementTree as ElementTree
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..animation import Group, Shape2D
from ..robjects.style import STROKE_WIDTH_SCALE

UNITS_PER_INCH = 1.0
"""Scene units per inch: the size an SVG keeps when none is asked for."""

_INCHES_PER_UNIT = {"": 1 / 96, "px": 1 / 96, "pt": 1 / 72, "pc": 1 / 6, "mm": 1 / 25.4, "cm": 1 / 2.54, "in": 1.0}
"""Inches per length unit of SVG (CSS units; a plain number is px)."""

Matrix = np.ndarray
"""A 3x3 affine matrix acting on column vectors ``(x, y, 1)``."""


@dataclass(frozen=True, slots=True)
class SVGPath:
    """One drawn element of an SVG: points in scene units (y up) and its
    paint. A color is ``None`` where the element has no fill or stroke."""

    points: np.ndarray
    fill_color: tuple[float, float, float] | None
    fill_opacity: float
    stroke_color: tuple[float, float, float] | None
    stroke_opacity: float
    stroke_width: float
    """In Shape2D stroke width units."""
    glyph: int | None = None
    """For a path drawn by a ``use`` of a ``symbol`` (how Typst and other
    tools place glyphs): which such use it is, in document order."""
    origin: tuple[float, float] | None = None
    """For such a path: where the use places the symbol's origin (a
    glyph's pen position on the baseline), in scene units."""

    def style(self) -> dict[str, Any]:
        """The Shape2D style arguments of this path."""
        style: dict[str, Any] = {
            "fill_opacity": self.fill_opacity if self.fill_color is not None else 0.0,
            "stroke_opacity": self.stroke_opacity if self.stroke_color is not None else 0.0,
            "stroke_width": self.stroke_width if self.stroke_color is not None else 0.0,
        }
        if self.fill_color is not None:
            style["fill_color"] = self.fill_color
        if self.stroke_color is not None:
            style["stroke_color"] = self.stroke_color
        if self.fill_color is not None and self.stroke_color is None:
            style["stroke_color"] = self.fill_color
        if self.stroke_color is not None and self.fill_color is None:
            style["fill_color"] = self.stroke_color
        return style


class SVG(Group):
    """A Group with one Shape2D per drawn element of an SVG document,
    centered at the origin.

    ``source`` is a file path or the SVG text itself. The drawing keeps
    its physical size unless ``width`` or ``height`` is given (in scene
    units; with both, ``height`` wins). Colors come from the document;
    ``style`` (any Shape2D style argument) overrides them on every shape.
    """

    def __init__(
        self,
        source: str | Path,
        width: float | None = None,
        height: float | None = None,
        name: str | None = None,
        **style: Any,
    ) -> None:
        text = str(source) if isinstance(source, str) and source.lstrip().startswith("<") else Path(source).read_text()
        paths = read_svg(text, width=width, height=height)
        super().__init__(*(Shape2D(path.points, self._path_name(path), **(path.style() | style)) for path in paths), name=name)

    def _path_name(self, path: SVGPath) -> str | None:
        """The name of the Shape2D of ``path`` (None: a generated one)."""
        return None

    @property
    def shapes(self) -> list[Shape2D]:
        """The Shape2D children, in document order."""
        return [child for child in self.children if isinstance(child, Shape2D)]


def read_svg(text: str, width: float | None = None, height: float | None = None, centered: bool = True) -> list[SVGPath]:
    """Every drawn element of the SVG document ``text``, centered at the
    origin and scaled as described in :class:`SVG`. With ``centered``
    false the document's origin stays the origin (y still points up)."""
    root = ElementTree.fromstring(text)
    by_id = {element.get("id"): element for element in root.iter() if element.get("id")}
    raw = list(_walk(root, _root_matrix(root), _Paint(gradients=_gradients(by_id)), by_id, itertools.count()))
    raw = [(points, paint) for points, paint in raw if len(points)]
    if not raw:
        return []
    everything = np.vstack([points for points, _ in raw])
    low, high = everything.min(axis=0), everything.max(axis=0)
    center = (low + high) / 2 if centered else np.zeros(2)
    size = high - low
    scale = UNITS_PER_INCH
    if height is not None and size[1] > 0:
        scale = height / size[1]
    elif width is not None and size[0] > 0:
        scale = width / size[0]
    flip = np.array([scale, -scale])
    return [paint.path((points - center) * flip, scale, center) for points, paint in raw]


# --------------------------------------------------------------------------
# Walking the document
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Paint:
    """The inherited presentation attributes of an element."""

    fill: str = "black"
    fill_opacity: float = 1.0
    stroke: str = "none"
    stroke_opacity: float = 1.0
    stroke_width: float = 1.0
    """In SVG user units of the element."""
    opacity: float = 1.0
    stroke_scale: float = 1.0
    """Inches per user unit at the element (for its stroke width)."""
    glyph: int | None = None
    """The symbol use the element is drawn by (SVGPath.glyph)."""
    origin: tuple[float, float] | None = None
    """Where that use places the symbol's origin, in inches with y down."""
    gradients: tuple[tuple[str, str], ...] = ()

    def inherit(self, element: ElementTree.Element, matrix: Matrix) -> _Paint:
        attributes = dict(element.attrib) | _style_declarations(element.get("style", ""))
        values: dict[str, Any] = {}
        for name, field in (("fill", "fill"), ("stroke", "stroke")):
            if name in attributes:
                values[field] = attributes[name].strip()
        for name, field in (("fill-opacity", "fill_opacity"), ("stroke-opacity", "stroke_opacity")):
            if name in attributes:
                values[field] = _number(attributes[name])
        if "stroke-width" in attributes:
            values["stroke_width"] = _length(attributes["stroke-width"])
        if "opacity" in attributes:
            values["opacity"] = self.opacity * _number(attributes["opacity"])
        # Stroke widths scale with the transform's mean linear factor.
        return replace(self, **values, stroke_scale=math.sqrt(abs(np.linalg.det(matrix[:2, :2]))))

    def path(self, points: np.ndarray, scale: float, center: np.ndarray) -> SVGPath:
        """The SVGPath of ``points`` (already moved and scaled), drawn
        ``scale`` scene units per inch around ``center`` (inches)."""
        fill, stroke = _paint_color(self.fill, self.gradients), _paint_color(self.stroke, self.gradients)
        width = self.stroke_width * self.stroke_scale * scale / STROKE_WIDTH_SCALE
        return SVGPath(
            points=points,
            fill_color=fill,
            fill_opacity=self.fill_opacity * self.opacity,
            stroke_color=stroke,
            stroke_opacity=self.stroke_opacity * self.opacity,
            stroke_width=width,
            glyph=self.glyph,
            origin=None if self.origin is None else (float((self.origin[0] - center[0]) * scale), float((center[1] - self.origin[1]) * scale)),
        )


_SKIPPED = {"defs", "symbol", "clipPath", "mask", "linearGradient", "radialGradient", "style", "title", "desc", "metadata", "text", "image", "filter", "pattern", "marker"}


def _walk(element: ElementTree.Element, matrix: Matrix, paint: _Paint, by_id: dict[str | None, ElementTree.Element], uses: Iterator[int]) -> Iterator[tuple[np.ndarray, _Paint]]:
    """``(points in inches with y down, paint)`` of every drawn element
    under ``element``."""
    attributes = _style_declarations(element.get("style", ""))
    if element.get("display", attributes.get("display")) == "none":
        return
    matrix = matrix @ _transform(element.get("transform", ""))
    paint = paint.inherit(element, matrix)
    tag = _tag(element)
    if tag == "use":
        target = by_id.get(_href(element))
        if target is None:
            return
        offset = _translation(_length(element.get("x", "0")), _length(element.get("y", "0")))
        if _tag(target) == "symbol":
            origin = _apply(matrix @ offset, np.zeros((1, 2)))[0]
            used = replace(paint, glyph=next(uses), origin=(float(origin[0]), float(origin[1])))
            yield from _walk_children(target, matrix @ offset, used, by_id, uses)
        else:
            yield from _walk(target, matrix @ offset, paint, by_id, uses)
        return
    if tag in _SKIPPED:
        return
    points = _element_points(element, tag)
    if points is not None and len(points):
        yield _apply(matrix, points), paint
    yield from _walk_children(element, matrix, paint, by_id, uses)


def _walk_children(element: ElementTree.Element, matrix: Matrix, paint: _Paint, by_id: dict[str | None, ElementTree.Element], uses: Iterator[int]) -> Iterator[tuple[np.ndarray, _Paint]]:
    for child in element:
        yield from _walk(child, matrix, paint, by_id, uses)


def _root_matrix(root: ElementTree.Element) -> Matrix:
    """From the root's user units to inches: the viewBox mapped onto the
    physical width and height (each axis on its own; no aspect ratio)."""
    view_box = [float(value) for value in re.split(r"[\s,]+", root.get("viewBox", "").strip()) if value]
    width = _inches(root.get("width"))
    height = _inches(root.get("height"))
    if len(view_box) != 4:
        return np.diag([_INCHES_PER_UNIT["px"], _INCHES_PER_UNIT["px"], 1.0])
    x, y, view_width, view_height = view_box
    scale_x = (width if width is not None else view_width * _INCHES_PER_UNIT["px"]) / view_width
    scale_y = (height if height is not None else view_height * _INCHES_PER_UNIT["px"]) / view_height
    return np.diag([scale_x, scale_y, 1.0]) @ _translation(-x, -y)


def _element_points(element: ElementTree.Element, tag: str) -> np.ndarray | None:
    get = lambda name: _length(element.get(name, "0"))  # noqa: E731
    if tag == "path":
        return path_points(element.get("d", ""))
    if tag == "rect":
        x, y, w, h = get("x"), get("y"), get("width"), get("height")
        rx = _length(element.get("rx", element.get("ry", "0")))
        ry = _length(element.get("ry", element.get("rx", "0")))
        if w <= 0 or h <= 0:
            return None
        rx, ry = min(rx, w / 2), min(ry, h / 2)
        if rx == 0 or ry == 0:
            return path_points(f"M{x},{y} h{w} v{h} h{-w} Z")
        return path_points(
            f"M{x + rx},{y} H{x + w - rx} A{rx},{ry} 0 0 1 {x + w},{y + ry} V{y + h - ry} "
            f"A{rx},{ry} 0 0 1 {x + w - rx},{y + h} H{x + rx} A{rx},{ry} 0 0 1 {x},{y + h - ry} "
            f"V{y + ry} A{rx},{ry} 0 0 1 {x + rx},{y} Z"
        )
    if tag in ("circle", "ellipse"):
        cx, cy = get("cx"), get("cy")
        rx = get("r") if tag == "circle" else get("rx")
        ry = get("r") if tag == "circle" else get("ry")
        if rx <= 0 or ry <= 0:
            return None
        return path_points(f"M{cx + rx},{cy} A{rx},{ry} 0 0 1 {cx},{cy + ry} A{rx},{ry} 0 0 1 {cx - rx},{cy} A{rx},{ry} 0 0 1 {cx},{cy - ry} A{rx},{ry} 0 0 1 {cx + rx},{cy} Z")
    if tag == "line":
        return path_points(f"M{get('x1')},{get('y1')} L{get('x2')},{get('y2')}")
    if tag in ("polyline", "polygon"):
        numbers = _NUMBER.findall(element.get("points", ""))
        if len(numbers) < 4:
            return None
        d = "M" + " ".join(numbers) + (" Z" if tag == "polygon" else "")
        return path_points(d)
    return None


# --------------------------------------------------------------------------
# Path data
# --------------------------------------------------------------------------

_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_TOKEN = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])|([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")
_ARGUMENTS = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}


def path_points(d: str) -> np.ndarray:
    """The points of SVG path data ``d`` (user units, y down) as cubic
    curves, four points each, ``(N, 2)``."""
    curves: list[tuple[np.ndarray, ...]] = []
    current = start = np.zeros(2)
    last_control: np.ndarray | None = None
    last_command = ""
    for command, values in _commands(d):
        upper = command.upper()
        relative = command.islower()
        base = current if relative else np.zeros(2)
        if upper == "M":
            current = start = base + values[:2]
            # Further pairs after a move are lines.
            for pair in values[2:].reshape(-1, 2):
                target = (current if relative else np.zeros(2)) + pair
                curves.append(_line(current, target))
                current = target
        elif upper in "LHV":
            if upper == "H":
                target = np.array([values[0] + (current[0] if relative else 0.0), current[1]])
            elif upper == "V":
                target = np.array([current[0], values[0] + (current[1] if relative else 0.0)])
            else:
                target = base + values
            curves.append(_line(current, target))
            current = target
        elif upper == "C":
            p1, p2, p3 = base + values[0:2], base + values[2:4], base + values[4:6]
            curves.append((current, p1, p2, p3))
            last_control, current = p2, p3
        elif upper == "S":
            p1 = 2 * current - last_control if last_command in "CS" and last_control is not None else current
            p2, p3 = base + values[0:2], base + values[2:4]
            curves.append((current, p1, p2, p3))
            last_control, current = p2, p3
        elif upper == "Q":
            control, end = base + values[0:2], base + values[2:4]
            curves.append(_quadratic(current, control, end))
            last_control, current = control, end
        elif upper == "T":
            control = 2 * current - last_control if last_command in "QT" and last_control is not None else current
            end = base + values
            curves.append(_quadratic(current, control, end))
            last_control, current = control, end
        elif upper == "A":
            end = base + values[5:7]
            curves.extend(_arc(current, values[0], values[1], values[2], bool(values[3]), bool(values[4]), end))
            current = end
        elif upper == "Z":
            if not np.allclose(current, start):
                curves.append(_line(current, start))
            current = start
        last_command = upper
    if not curves:
        return np.zeros((0, 2))
    return np.array([point for curve in curves for point in curve], dtype=np.float64)


def _commands(d: str) -> Iterator[tuple[str, np.ndarray]]:
    """``(command, its arguments)``, one per segment: repeated argument
    groups become repeated commands (a move's extra pairs stay with it)."""
    tokens = _tokens(d)
    index = 0
    command = ""
    while index < len(tokens):
        token = tokens[index]
        if isinstance(token, str):
            command = token
            index += 1
            if command.upper() == "Z":
                yield command, np.zeros(0)
                continue
        elif not command:
            raise ValueError(f"path data must start with a command: {d[:40]!r}")
        count = _ARGUMENTS[command.upper()]
        values = tokens[index : index + count]
        if len(values) < count or any(isinstance(value, str) for value in values):
            raise ValueError(f"path command {command!r} needs {count} numbers: {d[:40]!r}")
        index += count
        yield command, np.array(values, dtype=np.float64)
        # Implicit repetitions of a move are lines.
        if command == "M":
            command = "L"
        elif command == "m":
            command = "l"


def _tokens(d: str) -> list[str | float]:
    """Commands and numbers of ``d``. Arc flags may be written without
    separators (``a1 1 0 01 5 5``), so they are read one digit each."""
    tokens: list[str | float] = []
    position = 0
    command = ""
    argument = 0
    while position < len(d):
        character = d[position]
        if character in " \t\r\n,":
            position += 1
            continue
        if command.upper() == "A" and argument % 7 in (3, 4) and character in "01":
            tokens.append(float(character))
            argument += 1
            position += 1
            continue
        match = _TOKEN.match(d, position)
        if match is None:
            raise ValueError(f"cannot read path data at {d[position : position + 20]!r}")
        if match.group(1):
            command = match.group(1)
            argument = 0
            tokens.append(command)
        else:
            tokens.append(float(match.group(2)))
            argument += 1
        position = match.end()
    return tokens


def _line(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, ...]:
    return (start, (2 * start + end) / 3, (start + 2 * end) / 3, end)


def _quadratic(start: np.ndarray, control: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, ...]:
    return (start, start + 2 / 3 * (control - start), end + 2 / 3 * (control - end), end)


def _arc(start: np.ndarray, rx: float, ry: float, rotation: float, large: bool, sweep: bool, end: np.ndarray) -> list[tuple[np.ndarray, ...]]:
    """An elliptical arc as cubic curves of at most 90 degrees each
    (endpoint to center parameterization, SVG 1.1 appendix F.6,
    https://www.w3.org/TR/SVG11/implnote.html#ArcImplementationNotes)."""
    if np.allclose(start, end):
        return []
    rx, ry = abs(rx), abs(ry)
    if rx == 0 or ry == 0:
        return [_line(start, end)]
    phi = math.radians(rotation)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)
    dx, dy = (start - end) / 2
    x1 = cos_phi * dx + sin_phi * dy
    y1 = -sin_phi * dx + cos_phi * dy
    # Radii too small to reach the end are scaled up.
    radii_check = x1**2 / rx**2 + y1**2 / ry**2
    if radii_check > 1:
        rx, ry = rx * math.sqrt(radii_check), ry * math.sqrt(radii_check)
    numerator = rx**2 * ry**2 - rx**2 * y1**2 - ry**2 * x1**2
    factor = math.sqrt(max(0.0, numerator / (rx**2 * y1**2 + ry**2 * x1**2)))
    if large == sweep:
        factor = -factor
    cx1, cy1 = factor * rx * y1 / ry, -factor * ry * x1 / rx
    center = np.array([cos_phi * cx1 - sin_phi * cy1, sin_phi * cx1 + cos_phi * cy1]) + (start + end) / 2
    theta = math.atan2((y1 - cy1) / ry, (x1 - cx1) / rx)
    delta = math.atan2((-y1 - cy1) / ry, (-x1 - cx1) / rx) - theta
    if sweep and delta < 0:
        delta += math.tau
    elif not sweep and delta > 0:
        delta -= math.tau
    count = max(1, math.ceil(abs(delta) / (math.pi / 2) - 1e-9))
    step = delta / count
    handle = 4 / 3 * math.tan(step / 4)

    def point(angle: float) -> np.ndarray:
        x, y = rx * math.cos(angle), ry * math.sin(angle)
        return center + np.array([cos_phi * x - sin_phi * y, sin_phi * x + cos_phi * y])

    def tangent(angle: float) -> np.ndarray:
        x, y = -rx * math.sin(angle), ry * math.cos(angle)
        return np.array([cos_phi * x - sin_phi * y, sin_phi * x + cos_phi * y])

    curves = []
    previous = start
    for index in range(count):
        a, b = theta + index * step, theta + (index + 1) * step
        last = end if index == count - 1 else point(b)
        curves.append((previous, previous + handle * tangent(a), last - handle * tangent(b), last))
        previous = last
    return curves


# --------------------------------------------------------------------------
# Transforms, lengths and colors
# --------------------------------------------------------------------------


def _translation(x: float, y: float) -> Matrix:
    return np.array([[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]])


def _transform(text: str) -> Matrix:
    """The matrix of a ``transform`` attribute (functions applied left to right)."""
    matrix = np.eye(3)
    for name, arguments in re.findall(r"(\w+)\s*\(([^)]*)\)", text):
        values = [float(value) for value in _NUMBER.findall(arguments)]
        if name == "matrix" and len(values) == 6:
            a, b, c, d, e, f = values
            step = np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]])
        elif name == "translate":
            step = _translation(values[0], values[1] if len(values) > 1 else 0.0)
        elif name == "scale":
            step = np.diag([values[0], values[1] if len(values) > 1 else values[0], 1.0])
        elif name == "rotate":
            angle = math.radians(values[0])
            cos, sin = math.cos(angle), math.sin(angle)
            step = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
            if len(values) == 3:
                step = _translation(values[1], values[2]) @ step @ _translation(-values[1], -values[2])
        elif name == "skewX":
            step = np.array([[1.0, math.tan(math.radians(values[0])), 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        elif name == "skewY":
            step = np.array([[1.0, 0.0, 0.0], [math.tan(math.radians(values[0])), 1.0, 0.0], [0.0, 0.0, 1.0]])
        else:
            raise ValueError(f"unknown SVG transform {name!r}")
        matrix = matrix @ step
    return matrix


def _apply(matrix: Matrix, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:2, :2].T + matrix[:2, 2]


def _length(text: str) -> float:
    """A length in user units (units other than px are ignored there:
    user units already are the element's coordinate system)."""
    match = _NUMBER.match(text.strip())
    return float(match.group()) if match else 0.0


def _inches(text: str | None) -> float | None:
    if text is None:
        return None
    match = re.fullmatch(r"\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(px|pt|pc|mm|cm|in)?\s*", text)
    if match is None:
        return None  # percentages and the like: fall back to the viewBox
    return float(match.group(1)) * _INCHES_PER_UNIT[match.group(2) or ""]


def _number(text: str) -> float:
    text = text.strip()
    return float(text[:-1]) / 100 if text.endswith("%") else float(text)


def _style_declarations(text: str) -> dict[str, str]:
    declarations = (item.split(":", 1) for item in text.split(";") if ":" in item)
    return {name.strip(): value.strip() for name, value in declarations}


def _tag(element: ElementTree.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _href(element: ElementTree.Element) -> str | None:
    href = element.get("href") or element.get("{http://www.w3.org/1999/xlink}href")
    return href[1:] if href and href.startswith("#") else None


def _gradients(by_id: dict[str | None, ElementTree.Element]) -> tuple[tuple[str, str], ...]:
    """``(id, color of the first stop)`` of every gradient (following
    ``href`` to the gradient that has the stops)."""
    result = []
    for key, element in by_id.items():
        if _tag(element) not in ("linearGradient", "radialGradient"):
            continue
        source = element
        for _ in range(8):
            stops = [child for child in source if _tag(child) == "stop"]
            if stops or (target := by_id.get(_href(source))) is None:
                break
            source = target
        if stops:
            stop = stops[0]
            color = stop.get("stop-color") or _style_declarations(stop.get("style", "")).get("stop-color", "black")
            result.append((str(key), color))
    return tuple(result)


_NAMED_COLORS = {
    "black": "#000000", "white": "#ffffff", "red": "#ff0000", "green": "#008000", "lime": "#00ff00",
    "blue": "#0000ff", "yellow": "#ffff00", "cyan": "#00ffff", "aqua": "#00ffff", "magenta": "#ff00ff",
    "fuchsia": "#ff00ff", "gray": "#808080", "grey": "#808080", "silver": "#c0c0c0", "maroon": "#800000",
    "olive": "#808000", "navy": "#000080", "purple": "#800080", "teal": "#008080", "orange": "#ffa500",
}  # fmt: skip
"""The basic CSS color names (others raise)."""


def _paint_color(text: str, gradients: tuple[tuple[str, str], ...]) -> tuple[float, float, float] | None:
    """An SVG paint as an sRGB triple in [0, 1], ``None`` for ``none``."""
    text = text.strip()
    if text in ("none", "transparent", ""):
        return None
    if text.startswith("url("):
        key = text[4:].split(")", 1)[0].strip().strip("'\"").lstrip("#")
        return _paint_color(dict(gradients).get(key, "black"), ())
    text = _NAMED_COLORS.get(text.lower(), text)
    if text.startswith("#"):
        digits = text[1:]
        if len(digits) in (3, 4):
            digits = "".join(digit * 2 for digit in digits[:3])
        return tuple(int(digits[index : index + 2], 16) / 255 for index in (0, 2, 4))  # type: ignore[return-value]
    match = re.fullmatch(r"rgba?\(([^)]*)\)", text)
    if match:
        parts = [part.strip() for part in match.group(1).split(",")]
        return tuple(_number(part) if part.endswith("%") else float(part) / 255 for part in parts[:3])  # type: ignore[return-value]
    raise ValueError(f"unknown SVG color {text!r}")


__all__ = ["SVG", "UNITS_PER_INCH", "SVGPath", "path_points", "read_svg"]
