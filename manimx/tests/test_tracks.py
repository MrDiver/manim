"""Tracks: the per-signal composition rules of section 4.2 of the design doc."""

import math

import numpy as np
import pytest
import quaternion

from manimx.animation import DISCRETE, FACTOR, ROTATION, RootSignal
from manimx.animation.animations import SignalChange
from manimx.animation.rate_functions import linear, there_and_back
from manimx.animation.tracks import Change, Placement, Track

REL, ABS = Change.RELATIVE, Change.ABSOLUTE


class Formula:
    """A minimal custom animation: any object with a contribution method."""

    def __init__(self, function) -> None:
        self.function = function

    def contribution(self, signal, alpha, start_value):
        return self.function(alpha, start_value)


def place(track: Track, order: int, start: float, duration: float, mode: Change, value, rate=linear) -> Placement:
    placement = Placement(SignalChange(track.signal, mode, value), start, duration, order, rate)
    track.add(placement, mode)
    return placement


def track(base=0.0, kind=None, *changes: tuple) -> Track:
    result = Track(RootSignal(base, kind) if kind else RootSignal(base))
    for order, (start, duration, mode, value) in enumerate(changes):
        place(result, order, start, duration, mode, value)
    return result


def test_relative_changes_add_up() -> None:
    shifts = track(1.0, None, (0, 2, REL, 4.0), (1, 2, REL, -2.0))  # overlapping

    assert shifts.value_at(0.0) == 1.0
    assert shifts.value_at(1.0) == 3.0
    assert shifts.value_at(2.0) == 4.0  # +4 done, -2 halfway
    assert shifts.value_at(3.0) == shifts.final_value() == 3.0


def test_absolute_takes_over_smoothly_from_a_running_change() -> None:
    # A shift by +10 over [0, 10]; at t = 4 a move_to(0) over [4, 6] begins.
    moves = track(0.0, None, (0, 10, REL, 10.0), (4, 2, ABS, 0.0))

    assert moves.value_at(4.0) == pytest.approx(4.0)  # no jump: starts where the shift was
    assert moves.value_at(5.0) == pytest.approx(2.0)
    assert moves.value_at(8.0) == pytest.approx(0.0)  # the shift no longer counts


def test_later_absolute_wins_and_later_relative_adds_on_top() -> None:
    combined = track(0.0, None, (0, 1, ABS, 5.0), (0, 1, ABS, 9.0), (1, 1, REL, 1.0))

    assert combined.value_at(1.0) == 9.0
    assert combined.value_at(2.0) == 10.0


def test_boundaries_and_instant_changes() -> None:
    instant = track(False, DISCRETE, (2, 0, ABS, True))

    assert instant.value_at(1.999) is False
    assert instant.value_at(2.0) is True  # present from its start time on
    ramp = track(0.0, None, (1, 2, ABS, 1.0))
    assert ramp.value_at(1.0) == 0.0 and ramp.value_at(3.0) == 1.0 and ramp.value_at(99) == 1.0


def test_random_query_order_gives_identical_results() -> None:
    animated = track(0.0, None, (0, 3, REL, 3.0), (1, 2, ABS, -1.0), (2, 4, REL, 2.0))
    times = np.linspace(-1, 8, 37)

    forward = [animated.value_at(t) for t in times]
    shuffled = {t: animated.value_at(t) for t in np.random.default_rng(3).permutation(times)}
    assert forward == [shuffled[t] for t in times]


def test_rate_functions_shape_progress() -> None:
    signal = Track(RootSignal(0.0))
    place(signal, 0, 0, 2, REL, 1.0, rate=there_and_back)

    assert signal.value_at(1.0) == pytest.approx(1.0)
    assert signal.value_at(2.0) == pytest.approx(0.0)


def test_rotation_changes_compose_and_slerp() -> None:
    quarter = quaternion.from_rotation_vector([0, 0, math.pi / 2])
    spin = track(quaternion.one, ROTATION, (0, 1, REL, quarter), (1, 1, REL, quarter))

    np.testing.assert_allclose(quaternion.as_rotation_vector(spin.value_at(1.5)), [0, 0, 3 * math.pi / 4], atol=1e-9)


def test_factor_changes_multiply_geometrically() -> None:
    size = track(1.0, FACTOR, (0, 2, REL, 4.0))

    assert size.value_at(1.0) == pytest.approx(2.0)  # half of x4 is x2
    assert size.final_value() == pytest.approx(4.0)
    with pytest.raises(ValueError, match="positive"):
        track(1.0, FACTOR, (0, 1, REL, -1.0)).value_at(0.5)


def test_placements_must_arrive_in_authoring_order() -> None:
    animated = Track(RootSignal(0.0))
    place(animated, 5, 0, 1, REL, 1.0)

    with pytest.raises(ValueError, match="authoring order"):
        place(animated, 5, 0, 1, REL, 1.0)


def test_absolute_targets_are_validated_by_the_kind() -> None:
    animated = Track(RootSignal(0.0))
    placement = place(animated, 0, 0, 1, ABS, "3")

    assert animated.final_value() == 3.0 and placement.end == 1.0
    assert animated.end == 1.0 and Track(RootSignal(0.0)).end == 0.0


def test_animations_may_know_their_own_start() -> None:
    """Like FadeIn: an absolute animation that ignores V_e and starts at 0."""
    opacity = Track(RootSignal(1.0))
    opacity.add(Placement(Formula(lambda alpha, _start: alpha), 2, 2, 0, linear), ABS)

    assert opacity.value_at(1.0) == 1.0  # before the fade: untouched
    assert opacity.value_at(2.0) == 0.0  # the fade starts from 0, not from V_e
    assert opacity.value_at(3.0) == 0.5


def test_animations_receive_the_start_value() -> None:
    seen = []
    signal = Track(RootSignal(0.0))
    place(signal, 0, 0, 4, REL, 8.0)
    signal.add(Placement(Formula(lambda alpha, start: seen.append(start) or start), 2, 1, 1, linear), ABS)

    assert signal.value_at(3.0) == 4.0
    assert set(seen) == {4.0}


def test_one_placement_on_several_tracks_moves_together() -> None:
    """What a timeline editor does: drag one animation bar."""
    x, y = Track(RootSignal(0.0)), Track(RootSignal(0.0))
    shift = SignalChange(x.signal, REL, 2.0)

    class ShiftBoth:
        def contribution(self, signal, alpha, start_value):
            return 2.0 * alpha

    placement = Placement(ShiftBoth(), 0, 2, 0, linear)
    x.add(placement, REL)
    y.add(placement, REL)
    assert x.value_at(1.0) == y.value_at(1.0) == 1.0

    placement.start = 4.0  # both channels follow
    assert x.value_at(1.0) == y.value_at(1.0) == 0.0
    assert x.value_at(5.0) == y.value_at(5.0) == 1.0
    assert shift.contribution(x.signal, 0.5, 0.0) == 1.0


def test_refresh_recomputes_start_values_after_a_move() -> None:
    moves = Track(RootSignal(0.0))
    shift = place(moves, 0, 0, 10, REL, 10.0)
    place(moves, 1, 4, 2, ABS, 0.0)
    assert moves.value_at(4.0) == pytest.approx(4.0)

    shift.start = 2.0  # the shift now started later: at t = 4 it is at 2
    moves.refresh()

    assert moves.value_at(4.0) == pytest.approx(2.0)
