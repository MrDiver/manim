"""Monotone time maps: smooth, invertible curves from one time to another.

A clip's segments map global time to the scene's compiled time through
*knots*: the segment's start, cues synced to events, the edges of slow
motion ranges, the segment's end. Between knots the map is a monotone
cubic Hermite interpolation (PCHIP, Fritsch and Carlson 1980,
https://doi.org/10.1137/0717021):

* it passes exactly through every knot (a cue lands exactly on its word),
* it never overshoots and never runs backwards (time only moves forward),
* its slope, the playback speed, changes smoothly instead of jumping.

With only two knots it is a straight line, i.e. plain playback at a
constant scale.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence

import numpy as np


class MonotoneMap:
    """A strictly increasing function through ``(x, y)`` knots."""

    __slots__ = ("x", "y", "slopes")

    def __init__(self, x: Sequence[float], y: Sequence[float]) -> None:
        xs, ys = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if len(xs) < 2 or len(xs) != len(ys):
            raise ValueError("a time map needs at least two knots")
        if np.any(np.diff(xs) <= 0.0) or np.any(np.diff(ys) < 0.0):
            raise ValueError("time map knots must increase (time cannot run backwards)")
        self.x = xs
        self.y = ys
        self.slopes = _pchip_slopes(xs, ys)

    @property
    def start(self) -> float:
        return float(self.x[0])

    @property
    def end(self) -> float:
        return float(self.x[-1])

    def __call__(self, x: float) -> float:
        """The mapped value at ``x`` (clamped to the knot range)."""
        if x <= self.x[0]:
            return float(self.y[0])
        if x >= self.x[-1]:
            return float(self.y[-1])
        i = bisect.bisect_right(self.x, x) - 1
        x0, x1 = self.x[i], self.x[i + 1]
        h = x1 - x0
        s = (x - x0) / h
        # Cubic Hermite basis on [x0, x1].
        h00 = 2 * s**3 - 3 * s**2 + 1
        h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2
        h11 = s**3 - s**2
        return float(h00 * self.y[i] + h10 * h * self.slopes[i] + h01 * self.y[i + 1] + h11 * h * self.slopes[i + 1])

    def inverse(self, y: float) -> float:
        """The ``x`` that maps to ``y`` (bisection; the map is monotone)."""
        if y <= self.y[0]:
            return float(self.x[0])
        if y >= self.y[-1]:
            return float(self.x[-1])
        low, high = float(self.x[0]), float(self.x[-1])
        for _ in range(64):  # well below float resolution for any timeline
            middle = (low + high) / 2
            if self(middle) < y:
                low = middle
            else:
                high = middle
        return (low + high) / 2


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Knot slopes that keep a cubic Hermite interpolant monotone."""
    h = np.diff(x)
    delta = np.diff(y) / h
    slopes = np.zeros_like(y)
    if len(x) == 2:
        slopes[:] = delta[0]
        return slopes
    # Interior knots: weighted harmonic mean of the neighbouring secant
    # slopes, or zero where the data changes direction or is flat.
    for k in range(1, len(x) - 1):
        if delta[k - 1] * delta[k] <= 0.0:
            slopes[k] = 0.0
        else:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            slopes[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
    slopes[0] = _end_slope(h[0], h[1], delta[0], delta[1])
    slopes[-1] = _end_slope(h[-1], h[-2], delta[-1], delta[-2])
    return slopes


def _end_slope(h0: float, h1: float, d0: float, d1: float) -> float:
    """One-sided three-point estimate, limited to keep monotonicity."""
    slope = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if slope * d0 <= 0.0:
        return 0.0
    if d0 * d1 <= 0.0 and abs(slope) > abs(3 * d0):
        return 3 * d0
    return slope
