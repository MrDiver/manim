"""Tracks: the per-signal composition rules of section 4.2 of the design doc."""

import math

import numpy as np
import pytest
import quaternion

from manimx.animation import DISCRETE, FACTOR, ROTATION, RootSignal
from manimx.animation.rate_functions import linear, there_and_back
from manimx.animation.tracks import Change, Record, Track

REL, ABS = Change.RELATIVE, Change.ABSOLUTE


def track(base=0.0, kind=None, *records: tuple) -> Track:
    signal = RootSignal(base, kind) if kind else RootSignal(base)
    result = Track(signal)
    for order, (start, duration, mode, value) in enumerate(records):
        result.add(Record(order, start, duration, mode, value, rate=linear))
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


def test_boundaries_and_instant_records() -> None:
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
    signal.add(Record(0, 0, 2, REL, 1.0, rate=there_and_back))

    assert signal.value_at(1.0) == pytest.approx(1.0)
    assert signal.value_at(2.0) == pytest.approx(0.0)


def test_rotation_records_compose_and_slerp() -> None:
    quarter = quaternion.from_rotation_vector([0, 0, math.pi / 2])
    spin = track(quaternion.one, ROTATION, (0, 1, REL, quarter), (1, 1, REL, quarter))

    np.testing.assert_allclose(quaternion.as_rotation_vector(spin.value_at(1.5)), [0, 0, 3 * math.pi / 4], atol=1e-9)


def test_factor_records_multiply_geometrically() -> None:
    size = track(1.0, FACTOR, (0, 2, REL, 4.0))

    assert size.value_at(1.0) == pytest.approx(2.0)  # half of x4 is x2
    assert size.final_value() == pytest.approx(4.0)
    with pytest.raises(ValueError, match="positive"):
        track(1.0, FACTOR, (0, 1, REL, -1.0)).value_at(0.5)


def test_records_must_arrive_in_authoring_order() -> None:
    animated = Track(RootSignal(0.0))
    animated.add(Record(5, 0, 1, REL, 1.0))

    with pytest.raises(ValueError, match="authoring order"):
        animated.add(Record(5, 0, 1, REL, 1.0))


def test_absolute_targets_are_validated_by_the_kind() -> None:
    animated = Track(RootSignal(0.0))
    record = animated.add(Record(0, 0, 1, ABS, "3"))

    assert record.value == 3.0 and record.end == 1.0
    assert animated.end == 1.0 and Track(RootSignal(0.0)).end == 0.0
