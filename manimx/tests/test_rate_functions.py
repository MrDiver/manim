"""Rate functions ported from Manim: every one is usable as a record's rate."""

import inspect
import math

import numpy as np
import pytest
from manimx.userland import rate_functions as rates

# Rate functions that do not end at 1: they return to the start, or decay.
ENDS_ELSEWHERE = {"there_and_back", "there_and_back_with_pause", "wiggle", "running_start", "not_quite_there", "exponential_decay", "lingering"}
# Every function of the module whose first parameter is the progress `t`
# (wrappers like squish_rate_func take a function first), including linear
# and smooth, which the core defines and the module re-exports.
NAMES = sorted(
    name
    for name, function in inspect.getmembers(rates, inspect.isfunction)
    if function.__module__ in (rates.__name__, "manimx.animation.signals.rates") and list(inspect.signature(function).parameters)[:1] == ["t"]
)


@pytest.mark.parametrize("name", NAMES)
def test_rate_function_is_finite_and_starts_at_zero(name: str) -> None:
    function = getattr(rates, name)
    values = [float(function(t)) for t in np.linspace(0.0, 1.0, 41)]

    assert all(math.isfinite(v) and -1.0 <= v <= 2.0 for v in values)
    assert values[0] == pytest.approx(0.0, abs=1e-6)
    if name not in ENDS_ELSEWHERE:
        assert values[-1] == pytest.approx(1.0, abs=1e-6)


def test_wrappers_and_parameters() -> None:
    late = rates.squish_rate_func(rates.linear, 0.5, 1.0)
    assert late(0.25) == 0.0 and late(0.75) == pytest.approx(0.5) and late(1.0) == 1.0
    assert rates.squish_rate_func(rates.linear, 0.5, 0.5)(0.7) == 0.5  # zero-width window: its position
    assert rates.not_quite_there(rates.linear, 0.5)(1.0) == pytest.approx(0.5)
    assert rates.smooth(-1.0) == 0.0 and rates.smooth(2.0) == 1.0  # unit_interval clamps
    assert rates.there_and_back(0.5) == pytest.approx(1.0)
    assert rates.sigmoid(0.0) == 0.5
