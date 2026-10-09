"""Contributions as data: what an animation does to a signal, described
instead of computed.

``Animation.contribution(signal, alpha, start_value)`` defines an
animation. An animation that can also describe that function as one of
the values below returns it from :meth:`Animation.planned_contribution`;
the core (manimx._core, see CORE.md) then evaluates it without calling
Python. An animation that returns None is called back. Both forms must
give the same values (tests/test_plan.py checks the built-in ones).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


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


Contribution = To | By | Ramp | Power | Turn | Keyframes

__all__ = ["By", "Contribution", "Keyframes", "Power", "Ramp", "To", "Turn"]
