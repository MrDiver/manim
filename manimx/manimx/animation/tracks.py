"""Tracks: everything that happens to one root signal over time.

A track holds a signal's base value and the animation records that target
it. Its value at a local time ``t`` is a closed-form function of ``t``:
the records are folded in **authoring order** (section 4.2 of
docs/animation_engine_design.md):

    v = base
    for each record r with r.start <= t, in authoring order:
        a = rate(progress of r at t)
        relative r:  v = combine(v, scale(r.change, a))     # "by": adds up
        absolute r:  v = interpolate(V_r, r.target, a)      # "to": overrides

``V_r`` is the fold of all earlier records evaluated at ``r.start``: the
value the signal really has when ``r`` begins. Absolute animations
therefore start exactly where the signal is, even if an earlier animation
is still running, and from then on override it.

There is no replay from the start and no hidden state: evaluating a track
at any time, in any order, gives the same result.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .kinds import SignalKind
from .rate_functions import linear, smooth
from .signals import RootSignal

RateFunction = Callable[[float], float]


class Change(Enum):
    RELATIVE = "relative"
    """Changes the signal *by* an amount (shift, rotate by, scale by)."""
    ABSOLUTE = "absolute"
    """Changes the signal *to* a value (move to, set color)."""


@dataclass(slots=True, eq=False)
class Record:
    """One animation's effect on one signal, with absolute local times."""

    order: int
    """Position in authoring order; decides composition, not the start time."""
    start: float
    duration: float
    mode: Change
    value: Any
    """The relative change, or the absolute target."""
    rate: RateFunction = smooth
    start_value: Any = field(default=None, repr=False)
    """V_r for absolute records, filled in by the track (see module docs)."""

    @property
    def end(self) -> float:
        return self.start + self.duration

    def progress(self, t: float) -> float:
        """Rate-mapped progress at local time ``t`` (``t >= start``)."""
        if self.duration <= 0.0:
            return 1.0  # instantaneous: done from its start time on
        linear_progress = min(1.0, max(0.0, (t - self.start) / self.duration))
        return float(self.rate(linear_progress))


class Track:
    """A root signal's base value and its records."""

    __slots__ = ("signal", "base", "records")

    def __init__(self, signal: RootSignal) -> None:
        self.signal = signal
        self.base = signal.value
        self.records: list[Record] = []

    @property
    def kind(self) -> SignalKind:
        return self.signal.kind

    @property
    def end(self) -> float:
        return max((r.end for r in self.records), default=0.0)

    def add(self, record: Record) -> Record:
        """Append a record (records must arrive in authoring order)."""
        if self.records and record.order <= self.records[-1].order:
            raise ValueError("records must be added in authoring order")
        value = self.kind.validate(record.value) if record.mode is Change.ABSOLUTE else record.value
        record.value = value
        if record.mode is Change.ABSOLUTE:
            record.start_value = self._fold(record.start, len(self.records))
        self.records.append(record)
        return record

    def value_at(self, t: float) -> Any:
        return self._fold(t, len(self.records))

    def final_value(self) -> Any:
        """The value once every record has finished."""
        return self.value_at(self.end)

    def _fold(self, t: float, count: int) -> Any:
        """Fold the first ``count`` records at time ``t``."""
        kind = self.kind
        value = self.base
        for record in self.records[:count]:
            if record.start > t:
                continue
            alpha = record.progress(t)
            if record.mode is Change.RELATIVE:
                value = kind.combine(value, kind.scale(record.value, alpha))
            else:
                value = kind.interpolate(record.start_value, record.value, alpha)
        return value


__all__ = ["Change", "RateFunction", "Record", "Track", "linear", "smooth"]
