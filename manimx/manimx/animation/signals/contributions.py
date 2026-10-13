"""Contributions as data: what an animation does to a signal, described
instead of computed.

``Animation.contributions(alpha, start_values)`` defines an animation:
what it does to each of its targets. An animation that can also describe
that function for a target as one of the values below returns it from
:meth:`Animation.planned_contributions`; the core (manimx._core, see
CORE.md) then evaluates it without calling Python. Targets with None are
called back, all of an animation's in one call. Both forms must give the
same values (tests/test_plan.py checks the built-in ones).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .kinds import SignalKind
from .paths import partial_path


@dataclass(frozen=True, slots=True)
class To:
    """From the start value to ``value`` (``kind.interpolate``)."""

    value: Any


@dataclass(frozen=True, slots=True)
class By:
    """The fraction ``alpha`` of the relative change ``change`` (``kind.scale``)."""

    change: Any


@dataclass(frozen=True, slots=True)
class Ramp:
    """``base * alpha``, or ``base * (1 - alpha)`` when ``remaining``. The
    base is ``factor``, or the start value when ``factor`` is None.
    """

    factor: float | None
    remaining: bool = False


@dataclass(frozen=True, slots=True)
class Power:
    """``base ** alpha``."""

    base: float


@dataclass(frozen=True, slots=True)
class Turn:
    """A rotation by ``angle * alpha`` about ``axis``."""

    axis: tuple[float, float, float]
    angle: float


@dataclass(frozen=True, slots=True)
class Keyframes:
    """Recorded values at simulated ``times``, interpolated (a baked
    :class:`~.simulation.Simulation`).
    """

    run_time: float
    times: Sequence[float]
    values: Sequence[Any]


@dataclass(frozen=True, slots=True)
class Between:
    """From ``start`` to ``end`` (``kind.interpolate``) while the progress
    runs from ``lower`` to ``upper``: ``start`` before, ``end`` after. A
    window of no width (``lower == upper``) switches at ``upper``. Unlike
    :class:`To`, it ignores the signal's value when the animation starts.
    """

    start: Any
    end: Any
    lower: float = 0.0
    upper: float = 1.0


@dataclass(frozen=True, slots=True)
class Partial:
    """The part of the path ``points`` drawn so far: from its start to the
    fraction the progress has reached within ``lower`` to ``upper``
    (:func:`~.paths.partial_path`); backwards (from the full path to
    nothing) when ``reverse``.
    """

    points: Any
    lower: float = 0.0
    upper: float = 1.0
    reverse: bool = False


Contribution = To | By | Ramp | Power | Turn | Keyframes | Between | Partial


def window(alpha: float, lower: float, upper: float) -> float:
    """How far ``alpha`` is through ``lower`` to ``upper``, in [0, 1]."""
    if upper <= lower:
        return 1.0 if alpha >= upper else 0.0
    return min(1.0, max(0.0, (alpha - lower) / (upper - lower)))


def evaluate(planned: Between | Partial, kind: SignalKind, alpha: float) -> Any:
    """The value of the data forms that depend on nothing but themselves,
    in Python: what ``contribution()`` returns for animations defined by
    them (Morph, Create, Write).
    """
    progress = window(alpha, planned.lower, planned.upper)
    if isinstance(planned, Between):
        return kind.interpolate(planned.start, planned.end, progress)
    return partial_path(planned.points, 0.0, 1.0 - progress if planned.reverse else progress)

__all__ = ["Between", "By", "Contribution", "Keyframes", "Partial", "Power", "Ramp", "To", "Turn", "evaluate", "window"]
