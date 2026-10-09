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
from .audio import Sound, SoundEvent, mix, write_wav
from .camera import Camera, View
from .clipping_group import ClippingGroup
from .constants import (
    DEGREES,
    DL,
    DOWN,
    DR,
    IN,
    LEFT,
    ORIGIN,
    OUT,
    PI,
    RIGHT,
    TAU,
    UL,
    UP,
    UR,
)
from .curve3d import Curve3D, ParametricCurve3D
from .group import Group
from .kinds import COLOR, DISCRETE, FACTOR, PATH, ROTATION, SCALAR, SignalKind
from .light import DirectionalLight, Light, PointLight, SpotLight
from .node import Node, PositionMixin, RotationMixin, ScalableMixin, animatable
from .scene import SEGMENT_GAP, Cue, Emit, PreparedScene, Scene, Segment
from .shape2d import (
    Circle,
    Dot,
    Polyline,
    Rectangle,
    RegularPolygon,
    Shape2D,
    Square,
    Triangle,
)
from .shape3d import Box, ParametricSurface, Shape3D, Sphere, Torus
from .signal_tracks import Change, Entry, Placement, SignalTrack
from .signals import (
    DerivedSignal,
    RootSignal,
    Signal,
    TimeSignals,
    Vector3Signal,
    derive,
)
from .simulation import Simulation
from .snapshot import version_token
from .time_map import MonotoneMap
from .timing import (
    NEVER,
    Clip,
    InvalidCue,
    Ramp,
    SegmentTiming,
    Timeline,
    UnresolvedEvents,
)
from .worlds import World, World2D, World3D

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
    "Box",
    "Camera",
    "Change",
    "ChangeSignals",
    "Circle",
    "Clip",
    "ClippingGroup",
    "Cue",
    "Curve3D",
    "DerivedSignal",
    "DirectionalLight",
    "Dot",
    "Emit",
    "Entry",
    "FadeIn",
    "FadeOut",
    "Group",
    "GrowFromCenter",
    "Indicate",
    "InvalidCue",
    "Light",
    "MonotoneMap",
    "Node",
    "ParametricCurve3D",
    "ParametricSurface",
    "Placement",
    "PointLight",
    "Polyline",
    "PositionMixin",
    "PreparedScene",
    "Ramp",
    "Rectangle",
    "RegularPolygon",
    "RootSignal",
    "Rotate",
    "RotationMixin",
    "ScalableMixin",
    "Scene",
    "Segment",
    "SegmentTiming",
    "SetValue",
    "Shape2D",
    "Shape3D",
    "ShrinkToCenter",
    "Signal",
    "SignalChange",
    "SignalKind",
    "SignalTrack",
    "Simulation",
    "Sound",
    "SoundEvent",
    "Sphere",
    "SpotLight",
    "Square",
    "Target",
    "TimeSignals",
    "Timeline",
    "Torus",
    "Triangle",
    "UnresolvedEvents",
    "Vector3Signal",
    "View",
    "World",
    "World2D",
    "World3D",
    "animatable",
    "derive",
    "mix",
    "rate_functions",
    "version_token",
    "write_wav",
]
