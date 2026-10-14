"""The animation engine (core): signals, scene nodes, animations and
compiled scenes.

This package is the core of the scene side: the node classes with render
counterparts (Shape2D, Curve3D, Shape3D, the lights, Group,
ClippingGroup), the Animation contract, Scene, Timeline and Simulation.
What a project could write on top of it (Circle, FadeIn, rate functions,
the glTF loader, ...) lives in manimx.userland. It produces render state
(manimx.robjects) that any renderer can draw.

See docs/animation_engine_design.md.
"""

from .animations import (
    Animation,
    AnimationBuilder,
    AnimationGroup,
    ChangeSignals,
    LaggedStart,
    SetValue,
    SignalChange,
    Succession,
    Target,
)
from .audio import Sound, SoundEvent, mix, write_wav
from .nodes.camera import Camera
from .nodes.camera_view import CameraView
from .nodes.clipping_group import ClippingGroup
from .nodes.curve3d import Curve3D
from .nodes.group import Group
from .nodes.light import DirectionalLight, Light, PointLight, SpotLight
from .nodes.node import Node, PositionMixin, RotationMixin, ScalableMixin, animatable
from .nodes.particles import Particles
from .nodes.shape2d import Shape2D
from .nodes.shape3d import Shape3D
from .scene import SEGMENT_GAP, Cue, Emit, PreparedScene, Scene, Segment
from .signals.kinds import (
    ARRAY,
    COLOR,
    COLORS,
    DISCRETE,
    FACTOR,
    PATH,
    ROTATION,
    SCALAR,
    SCALARS,
    SignalKind,
)
from .signals.rates import linear, smooth
from .signals.signal import (
    DerivedSignal,
    RootSignal,
    Signal,
    TimeSignals,
    Vector3Signal,
    derive,
)
from .signals.signal_tracks import Change, Entry, Placement, SignalTrack
from .simulation import Simulation
from .time_map import MonotoneMap
from .timing import (
    NEVER,
    AudioClip,
    Clip,
    InvalidCue,
    Ramp,
    SegmentTiming,
    Timeline,
    Track,
    UnresolvedEvents,
)

__all__ = [
    "ARRAY",
    "COLOR",
    "COLORS",
    "DISCRETE",
    "FACTOR",
    "NEVER",
    "PATH",
    "ROTATION",
    "SCALAR",
    "SCALARS",
    "SEGMENT_GAP",
    "Animation",
    "AnimationBuilder",
    "AnimationGroup",
    "AudioClip",
    "Track",
    "Camera",
    "CameraView",
    "Change",
    "ChangeSignals",
    "Clip",
    "ClippingGroup",
    "Cue",
    "Curve3D",
    "DerivedSignal",
    "DirectionalLight",
    "Emit",
    "Entry",
    "Group",
    "InvalidCue",
    "LaggedStart",
    "Light",
    "MonotoneMap",
    "Node",
    "Particles",
    "Placement",
    "PointLight",
    "PositionMixin",
    "PreparedScene",
    "Ramp",
    "RootSignal",
    "RotationMixin",
    "ScalableMixin",
    "Scene",
    "Segment",
    "SegmentTiming",
    "SetValue",
    "Shape2D",
    "Shape3D",
    "Signal",
    "SignalChange",
    "SignalKind",
    "SignalTrack",
    "Simulation",
    "Sound",
    "SoundEvent",
    "SpotLight",
    "Succession",
    "Target",
    "TimeSignals",
    "Timeline",
    "UnresolvedEvents",
    "Vector3Signal",
    "animatable",
    "derive",
    "linear",
    "mix",
    "smooth",
    "write_wav",
]
