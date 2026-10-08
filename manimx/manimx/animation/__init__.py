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
from .scene import SEGMENT_GAP, Cue, Emit, PreparedScene, Scene, Segment
from .signals import DerivedSignal, RootSignal, Signal, TimeSignals, Vector3Signal, derive
from .snapshot import version_token
from .time_map import MonotoneMap
from .timing import NEVER, Clip, InvalidCue, Ramp, SegmentTiming, Timeline, UnresolvedEvents
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
    "Cue",
    "DerivedSignal",
    "Dot",
    "Emit",
    "Entry",
    "FadeIn",
    "FadeOut",
    "Group",
    "GrowFromCenter",
    "Indicate",
    "InvalidCue",
    "Mobject",
    "MonotoneMap",
    "Placement",
    "Polyline",
    "PreparedScene",
    "Ramp",
    "Rectangle",
    "RegularPolygon",
    "RootSignal",
    "Rotate",
    "Scene",
    "Segment",
    "SegmentTiming",
    "SetValue",
    "ShrinkToCenter",
    "Signal",
    "SignalChange",
    "SignalKind",
    "SignalTrack",
    "Square",
    "Target",
    "TimeSignals",
    "Timeline",
    "Triangle",
    "UnresolvedEvents",
    "VMobject",
    "Vector3Signal",
    "animatable",
    "derive",
    "rate_functions",
    "version_token",
]
