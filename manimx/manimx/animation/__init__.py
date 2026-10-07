"""The animation engine: signals, animations and compiled scene clips.

See docs/animation_engine_design.md. This package never imports the
renderer; it produces render state (manimx.objects) that any renderer can
draw.
"""

from .kinds import COLOR, DISCRETE, PATH, ROTATION, SCALAR, SignalKind
from .signals import DerivedSignal, RootSignal, Signal, TimeSignals, Vector3Signal, derive

__all__ = [
    "COLOR",
    "DISCRETE",
    "PATH",
    "ROTATION",
    "SCALAR",
    "DerivedSignal",
    "RootSignal",
    "Signal",
    "SignalKind",
    "TimeSignals",
    "Vector3Signal",
    "derive",
]
