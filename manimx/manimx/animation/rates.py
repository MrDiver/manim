"""The two rate functions the animation system itself uses as defaults:
``smooth`` (every Animation) and ``linear`` (simulations, instant changes).

Every other rate function is userland (manimx.userland.rate_functions),
which re-exports these two.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Protocol

import numpy as np


def sigmoid(x: float) -> float:
    """Logistic function (ported from manim.utils.simple_functions)."""
    return 1.0 / (1 + np.exp(-x))


# TODO: rewrite this to use ParamSpec when Python 3.9 is out of life
class RateFunction(Protocol):
    def __call__(self, t: float, *args: Any, **kwargs: Any) -> float: ...


# This is a decorator that makes sure any function it's used on will
# return 0 if t<0 and 1 if t>1.
def unit_interval(function: RateFunction) -> RateFunction:
    @wraps(function)
    def wrapper(t: float, *args: Any, **kwargs: Any) -> float:
        if 0 <= t <= 1:
            return function(t, *args, **kwargs)
        elif t < 0:
            return 0
        else:
            return 1

    return wrapper


# This is a decorator that makes sure any function it's used on will
# return 0 if t<0 or t>1.
def zero(function: RateFunction) -> RateFunction:
    @wraps(function)
    def wrapper(t: float, *args: Any, **kwargs: Any) -> float:
        if 0 <= t <= 1:
            return function(t, *args, **kwargs)
        else:
            return 0

    return wrapper


@unit_interval
def linear(t: float) -> float:
    return t


@unit_interval
def smooth(t: float, inflection: float = 10.0) -> float:
    error = sigmoid(-inflection / 2)
    return min(
        max((sigmoid(inflection * (t - 0.5)) - error) / (1 - 2 * error), 0),
        1,
    )


__all__ = ["RateFunction", "linear", "sigmoid", "smooth", "unit_interval", "zero"]
