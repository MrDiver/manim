"""Signal tracks: how the animations touching one root signal combine over time.

(Not to be confused with timeline rows in an editor, which are called
lanes; see the glossary in docs/animation_engine_design.md.)

Animations are the units on the timeline: objects that reference their
node and are evaluated with a progress ``alpha``. Two small structures
connect them to signals:

* a :class:`Placement` is one animation placed on the timeline: the
  animation reference plus its start, duration, rate and authoring order.
  It is what a timeline editor shows as a bar and lets the user drag.
* an :class:`Entry` is a placement's appearance on one signal's track: how
  the animation changes that signal (by an amount, or to a value) and the
  signal's value when the animation starts. One placement has one entry per
  signal it touches (``shift(UR)`` touches x and y).

A :class:`SignalTrack` holds one root signal's base value and its entries. The
value at a local time ``t`` is a closed-form function of ``t``: the entries
are folded in **authoring order** (section 4.2 of
docs/animation_engine_design.md):

    v = base
    for each entry e whose placement p has p.start <= t, in authoring order:
        a = p.rate(progress of p at t)
        c = p.animation.contribution(signal, a, V_e)
        relative e:  v = combine(v, c)    # "by": adds up
        absolute e:  v = c                # "to": overrides

``V_e`` is the fold of all earlier entries evaluated at ``p.start``: the
value the signal really has when the animation begins. "To" animations such
as ``move_to`` interpolate from it, so they start exactly where the signal
is, even while an earlier animation is still running, and override it from
then on. Animations that know their own start (``FadeIn`` goes from 0)
simply ignore it. ``V_e`` depends on the placements of earlier entries, so
it is cached and recomputed by :meth:`SignalTrack.refresh` after a placement
changes.

There is no replay from the start and no hidden state: evaluating a track
at any time, in any order, gives the same result, provided animations'
contributions are pure (a documented contract for custom animations).

Tracks do not import the animation classes: they only need an object with
a ``contribution`` method (:class:`Contributor`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from .kinds import SignalKind
from .rates import linear, smooth
from .signals import RootSignal

RateFunction = Callable[[float], float]
Contribution = Callable[[float, Any], Any]
"""``contribution(alpha, start_value)``: a relative change, or an absolute value."""


class Change(Enum):
    RELATIVE = "relative"
    """Changes the signal *by* an amount (shift, rotate by, scale by)."""
    ABSOLUTE = "absolute"
    """Changes the signal *to* a value (move to, set color, fade in)."""


class Contributor(Protocol):
    """What a track needs from an animation."""

    def contribution(self, signal: RootSignal, alpha: float, start_value: Any) -> Any:
        """The relative change (or absolute value) of ``signal`` at progress
        ``alpha``; ``start_value`` is the signal's value when the animation
        starts."""
        ...


@dataclass(eq=False)
class Placement:
    """One animation placed on the timeline (local scene time)."""

    animation: Contributor
    start: float
    duration: float
    order: int
    """Position in authoring order; decides composition, not the start time."""
    rate: RateFunction = smooth

    @property
    def end(self) -> float:
        return self.start + self.duration

    def progress(self, t: float) -> float:
        """Rate-mapped progress at local time ``t`` (``t >= start``)."""
        if self.duration <= 0.0:
            return 1.0  # instantaneous: done from its start time on
        # At and after the end exactly 1: (end - start) / duration can round
        # to just below 1, and a switch at the end (Between with lower ==
        # upper == 1) would then never happen.
        linear_progress = 1.0 if t >= self.end else min(1.0, max(0.0, (t - self.start) / self.duration))
        return float(self.rate(linear_progress))


@dataclass(eq=False)
class Entry:
    """A placement's appearance on one signal's track."""

    placement: Placement
    mode: Change
    start_value: Any = field(default=None, repr=False)
    """The signal's value when the animation starts (V_e, see module docs)."""


class SignalTrack:
    """A root signal's base value and the entries of the animations on it."""

    __slots__ = ("signal", "base", "entries")

    def __init__(self, signal: RootSignal) -> None:
        self.signal = signal
        self.base = signal.value
        self.entries: list[Entry] = []

    @property
    def kind(self) -> SignalKind:
        return self.signal.kind

    @property
    def end(self) -> float:
        return max((e.placement.end for e in self.entries), default=0.0)

    def add(self, placement: Placement, mode: Change) -> Entry:
        """Add ``placement`` to this track (placements arrive in authoring order)."""
        if self.entries and placement.order <= self.entries[-1].placement.order:
            raise ValueError("placements must be added in authoring order")
        entry = Entry(placement, mode)
        entry.start_value = self._fold(placement.start, len(self.entries))
        self.entries.append(entry)
        return entry

    def refresh(self) -> None:
        """Recompute every entry's start value after placements changed."""
        for index, entry in enumerate(self.entries):
            entry.start_value = self._fold(entry.placement.start, index)

    def value_at(self, t: float) -> Any:
        return self._fold(t, len(self.entries))

    def final_value(self) -> Any:
        """The value once every animation on this track has finished."""
        return self.value_at(self.end)

    def _fold(self, t: float, count: int) -> Any:
        """Fold the first ``count`` entries at time ``t``."""
        kind = self.kind
        value = self.base
        for entry in self.entries[:count]:
            placement = entry.placement
            if placement.start > t:
                continue
            contribution = placement.animation.contribution(self.signal, placement.progress(t), entry.start_value)
            if entry.mode is Change.RELATIVE:
                value = kind.combine(value, contribution)
            else:
                value = kind.validate(contribution)
        return value


__all__ = ["Change", "Contributor", "Entry", "Placement", "RateFunction", "SignalTrack", "linear", "smooth"]
