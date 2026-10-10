"""Userland: everything a project could have written itself on top of the
core (manimx.animation, manimx.robjects).

Shapes built on the core node classes, the built-in animations, rate
functions, direction constants, geometry generators, the glTF loader,
the SVG reader and Typst text.
Nothing in the core imports this package, so it can be changed, replaced
or extended without touching how animation and rendering work.
"""

from . import geometry, rate_functions
from .animations import Create, DrawBorderThenFill, FadeIn, FadeOut, GrowFromCenter, Indicate, Morph, MorphMatching, Rotate, ShrinkToCenter, Uncreate, Write
from .constants import DEGREES, DL, DOWN, DR, IN, LEFT, ORIGIN, OUT, PI, RIGHT, TAU, UL, UP, UR
from .curves3d import ParametricCurve3D
from .gltf import load_gltf
from .shapes2d import Circle, Dot, Polyline, Rectangle, RegularPolygon, Square, Triangle
from .shapes3d import Box, ParametricSurface, Sphere, Torus
from .svg import SVG, read_svg
from .typst import Typst, TypstMath

__all__ = [
    "DEGREES",
    "DL",
    "DOWN",
    "DR",
    "IN",
    "LEFT",
    "ORIGIN",
    "OUT",
    "PI",
    "RIGHT",
    "TAU",
    "UL",
    "UP",
    "UR",
    "Box",
    "Circle",
    "Create",
    "Dot",
    "DrawBorderThenFill",
    "FadeIn",
    "FadeOut",
    "GrowFromCenter",
    "Indicate",
    "Morph",
    "MorphMatching",
    "ParametricCurve3D",
    "ParametricSurface",
    "Polyline",
    "Rectangle",
    "RegularPolygon",
    "Rotate",
    "SVG",
    "ShrinkToCenter",
    "Sphere",
    "Square",
    "Torus",
    "Triangle",
    "Typst",
    "TypstMath",
    "Uncreate",
    "Write",
    "geometry",
    "load_gltf",
    "rate_functions",
    "read_svg",
]
