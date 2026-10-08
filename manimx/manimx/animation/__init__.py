"""The animation engine: signals, animations and compiled scenes.

See docs/animation_engine_design.md. This package never imports the
renderer; it produces render state (manimx.objects) that any renderer can
draw.
"""

from . import rate_functions
from .animations import (
    Animation,
    AnimationBuilder,
    ChangeSignals,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    Indicate,
    Rotate,
    SetValue,
    ShrinkToCenter,
    SignalChange,
    Target,
)
from .constants import DEGREES, DL, DOWN, DR, IN, LEFT, ORIGIN, OUT, PI, RIGHT, TAU, UL, UP, UR
from .kinds import COLOR, DISCRETE, FACTOR, PATH, ROTATION, SCALAR, SignalKind
from .mobjects import (
    Circle,
    ClippingGroup,
    Dot,
    Group,
    Mobject,
    Polyline,
    Rectangle,
    RegularPolygon,
    Square,
    Triangle,
    VMobject,
    animatable,
)
from .scene import PreparedScene, Scene
from .signals import DerivedSignal, RootSignal, Signal, TimeSignals, Vector3Signal, derive
from .snapshot import version_token
from .tracks import Change, Entry, Placement, Track

__all__ = [
    "COLOR",
    "DEGREES",
    "DISCRETE",
    "DL",
    "DOWN",
    "DR",
    "FACTOR",
    "IN",
    "LEFT",
    "ORIGIN",
    "OUT",
    "PATH",
    "PI",
    "RIGHT",
    "ROTATION",
    "SCALAR",
    "TAU",
    "UL",
    "UP",
    "UR",
    "Animation",
    "AnimationBuilder",
    "Change",
    "ChangeSignals",
    "Circle",
    "ClippingGroup",
    "DerivedSignal",
    "Dot",
    "Entry",
    "FadeIn",
    "FadeOut",
    "Group",
    "GrowFromCenter",
    "Indicate",
    "Mobject",
    "Placement",
    "Polyline",
    "PreparedScene",
    "Rectangle",
    "RegularPolygon",
    "RootSignal",
    "Rotate",
    "Scene",
    "SetValue",
    "ShrinkToCenter",
    "Signal",
    "SignalChange",
    "SignalKind",
    "Square",
    "Target",
    "TimeSignals",
    "Track",
    "Triangle",
    "VMobject",
    "Vector3Signal",
    "animatable",
    "derive",
    "rate_functions",
    "version_token",
]
