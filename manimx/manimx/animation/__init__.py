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
from .scene import SEGMENT_GAP, Emit, PreparedScene, Scene, Segment
from .signals import DerivedSignal, RootSignal, Signal, TimeSignals, Vector3Signal, derive
from .snapshot import version_token
from .timing import NEVER, Clip, Timeline, UnresolvedEvents
from .signal_tracks import Change, Entry, Placement, SignalTrack

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
    "NEVER",
    "ORIGIN",
    "OUT",
    "PATH",
    "PI",
    "RIGHT",
    "ROTATION",
    "SCALAR",
    "SEGMENT_GAP",
    "TAU",
    "UL",
    "UP",
    "UR",
    "Animation",
    "AnimationBuilder",
    "Change",
    "ChangeSignals",
    "Circle",
    "Clip",
    "ClippingGroup",
    "DerivedSignal",
    "Dot",
    "Emit",
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
    "Segment",
    "SetValue",
    "ShrinkToCenter",
    "Signal",
    "SignalChange",
    "SignalKind",
    "Square",
    "Target",
    "TimeSignals",
    "Timeline",
    "SignalTrack",
    "Triangle",
    "UnresolvedEvents",
    "VMobject",
    "Vector3Signal",
    "animatable",
    "derive",
    "rate_functions",
    "version_token",
]
