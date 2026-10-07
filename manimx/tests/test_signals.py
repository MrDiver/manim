"""Signal core: kinds, propagation, versions, cycles, subscriptions."""

import math

import numpy as np
import pytest
import quaternion

from manimx.animation import (
    COLOR,
    DISCRETE,
    PATH,
    ROTATION,
    SCALAR,
    DerivedSignal,
    RootSignal,
    TimeSignals,
    Vector3Signal,
    derive,
)

# --- kinds ------------------------------------------------------------------------


def test_scalar_kind() -> None:
    assert SCALAR.interpolate(1.0, 3.0, 0.25) == 1.5
    assert SCALAR.combine(1.0, SCALAR.scale(4.0, 0.5)) == 3.0
    assert SCALAR.validate("2") == 2.0


def test_rotation_kind_takes_the_shortest_arc() -> None:
    quarter = quaternion.from_rotation_vector([0, 0, math.pi / 2])

    halfway = ROTATION.interpolate(quaternion.one, -quarter, 0.5)  # -q is the same rotation

    np.testing.assert_allclose(quaternion.as_rotation_vector(halfway), [0, 0, math.pi / 4], atol=1e-12)
    composed = ROTATION.combine(quarter, ROTATION.scale(quarter, 1.0))
    np.testing.assert_allclose(quaternion.as_rotation_vector(composed), [0, 0, math.pi / 2 * 2], atol=1e-12)
    assert ROTATION.validate(quaternion.quaternion(2, 0, 0, 0)) == quaternion.one
    with pytest.raises(TypeError):
        ROTATION.validate((1, 0, 0, 0))
    with pytest.raises(ValueError, match="zero"):
        ROTATION.validate(quaternion.quaternion(0, 0, 0, 0))


def test_color_kind() -> None:
    assert COLOR.interpolate((0, 0, 0), (1, 0.5, 0), 0.5) == (0.5, 0.25, 0.0)
    assert COLOR.combine((0.9, 0.5, 0.1), (0.3, 0.0, -0.5)) == (1.0, 0.5, 0.0)  # clamped
    assert COLOR.scale((0.2, 0.4, 0.6), 0.5) == pytest.approx((0.1, 0.2, 0.3))
    with pytest.raises(ValueError, match="channels"):
        COLOR.validate((1, 0))
    with pytest.raises(ValueError, match="channel counts"):
        COLOR.interpolate((0, 0, 0), (0, 0, 0, 1), 0.5)


def test_path_kind() -> None:
    a = PATH.validate([[0, 0], [1, 1]])
    b = PATH.validate([[2, 0], [3, 1]])

    np.testing.assert_allclose(PATH.interpolate(a, b, 0.5), [[1, 0], [2, 1]])
    np.testing.assert_allclose(PATH.combine(a, PATH.scale(np.array([2.0, 0.0]), 0.5)), [[1, 0], [2, 1]])
    assert not a.flags.writeable
    assert PATH.equal(a, PATH.validate([[0, 0], [1, 1]])) and not PATH.equal(a, b)
    with pytest.raises(ValueError, match="align them first"):
        PATH.interpolate(a, PATH.validate([[0, 0]]), 0.5)
    with pytest.raises(ValueError, match="2D point array"):
        PATH.validate([1, 2, 3])


def test_discrete_kind_switches_at_the_end() -> None:
    assert DISCRETE.interpolate("a", "b", 0.99) == "a"
    assert DISCRETE.interpolate("a", "b", 1.0) == "b"
    with pytest.raises(TypeError, match="relative"):
        DISCRETE.combine(True, False)
    with pytest.raises(TypeError, match="relative"):
        DISCRETE.scale(True, 0.5)
    array = np.zeros(3)
    assert DISCRETE.equal(array, array) and not DISCRETE.equal(array, np.zeros(3))


# --- root signals -----------------------------------------------------------------


def test_root_signal_versions_count_real_changes() -> None:
    x = RootSignal(1.0, name="x")

    assert x.version == 0
    assert x.set(2.0) is True and x.version == 1
    assert x.set(2.0) is False and x.version == 1  # same value: nothing happens
    x.value = 3
    assert x.value == 3.0 and x.version == 2
    assert repr(x) == "RootSignal('x', scalar)"


def test_signal_ids_are_unique() -> None:
    assert len({RootSignal(0.0).id for _ in range(100)}) == 100


# --- derived signals --------------------------------------------------------------


def test_derived_signals_follow_their_inputs() -> None:
    x, y = RootSignal(1.0), RootSignal(2.0)
    total = derive(lambda a, b: a + b, x, y)
    doubled = derive(lambda t: 2 * t, total)

    assert doubled.value == 6.0
    x.set(10.0)
    assert total.value == 12.0 and doubled.value == 24.0


def test_derived_signals_are_lazy_and_compute_once() -> None:
    calls = []
    x = RootSignal(1.0)
    y = derive(lambda a: calls.append(a) or a * 2, x)

    for value in (2.0, 3.0, 4.0):
        x.set(value)
    assert calls == []  # nothing read yet
    assert y.value == 8.0 and y.value == 8.0
    assert calls == [4.0]  # computed once, with the latest input


def test_diamond_dependencies_compute_each_signal_once() -> None:
    calls = {"b": 0, "c": 0, "d": 0}
    a = RootSignal(1.0)

    def count(name, function):
        def wrapped(*values):
            calls[name] += 1
            return function(*values)

        return wrapped

    b = derive(count("b", lambda v: v + 1), a)
    c = derive(count("c", lambda v: v * 2), a)
    d = derive(count("d", lambda u, v: u + v), b, c)

    assert d.value == 4.0
    a.set(2.0)
    assert d.value == 7.0
    assert calls == {"b": 2, "c": 2, "d": 2}


def test_derived_version_only_changes_with_the_value() -> None:
    x = RootSignal(1.0)
    sign = derive(lambda v: 1.0 if v > 0 else -1.0, x)

    assert sign.version == 1
    x.set(5.0)
    assert sign.version == 1  # recomputed, same value
    x.set(-5.0)
    assert sign.version == 2


def test_derived_signal_validates_with_its_kind() -> None:
    x = RootSignal(1.0)
    path = derive(lambda v: [[v, 0.0], [v, 1.0]], x, kind=PATH)

    assert not path.value.flags.writeable
    np.testing.assert_allclose(path.value, [[1, 0], [1, 1]])


def test_rebind_switches_inputs_and_rejects_cycles() -> None:
    a, b = RootSignal(1.0), RootSignal(10.0)
    follower = derive(lambda v: v + 0.5, a)
    downstream = derive(lambda v: v * 2, follower)
    assert downstream.value == 3.0

    follower.rebind(lambda v: v - 0.5, (b,))
    assert downstream.value == 19.0
    a.set(100.0)
    assert downstream.value == 19.0  # no longer listening to a
    assert a.dependents and not list(a.dependents())

    with pytest.raises(ValueError, match="cycle"):
        follower.rebind(lambda v: v, (downstream,))
    with pytest.raises(ValueError, match="cycle"):
        follower.rebind(lambda v: v, (follower,))
    assert follower.inputs == (b,)


def test_dependents_are_transitive() -> None:
    a = RootSignal(0.0)
    b = derive(lambda v: v, a)
    c = derive(lambda v: v, b)

    assert {s.id for s in a.dependents()} == {b.id, c.id}


# --- subscriptions ----------------------------------------------------------------


def test_subscribers_are_told_about_changes() -> None:
    x = RootSignal(1.0)
    y = derive(lambda v: v + 1, x)
    seen = []
    unsubscribe_x = x.subscribe(lambda s: seen.append(("x", s.value)))
    y.subscribe(lambda s: seen.append(("y", s.value)))
    y.value  # noqa: B018 - a derived signal notifies once it has been read

    x.set(2.0)
    x.set(2.0)  # unchanged: no notification
    unsubscribe_x()
    unsubscribe_x()  # unsubscribing twice is harmless
    x.set(3.0)

    assert seen == [("x", 2.0), ("y", 3.0), ("y", 4.0)]


def test_derived_notifies_once_until_read_again() -> None:
    x = RootSignal(1.0)
    y = derive(lambda v: v, x)
    calls = []
    y.subscribe(lambda s: calls.append(1))
    y.value  # noqa: B018

    x.set(2.0)
    x.set(3.0)  # still stale: no second notification
    assert len(calls) == 1
    y.value  # noqa: B018
    x.set(4.0)
    assert len(calls) == 2


# --- vectors and time -------------------------------------------------------------


def test_vector_signal_has_independent_axes() -> None:
    position = Vector3Signal((1.0, 2.0, 3.0), name="position")
    label_x = derive(lambda x: x + 0.5, position.x)

    assert position.value == (1.0, 2.0, 3.0)
    assert position.set((1.0, 5.0, 3.0)) is True
    assert position.version == (0, 1, 0)  # only y changed
    position.value = (4.0, 5.0, 3.0)
    assert label_x.value == 4.5
    assert position.set((4.0, 5.0, 3.0)) is False
    assert position.x.name == "position.x"


def test_time_signals() -> None:
    times = TimeSignals()
    spin = derive(lambda t: t * 2.0, times.global_time)
    progress = derive(lambda t: t / 10.0, times.scene_time)

    times.set(scene_time=4.0, global_time=12.0)

    assert spin.value == 24.0 and progress.value == 0.4
    times.set(scene_time=4.0, global_time=13.0)  # a hold: scene time stands still
    assert progress.version == 1 and spin.value == 26.0


def test_derived_signal_class_can_be_used_directly() -> None:
    x = RootSignal(2.0)
    squared = DerivedSignal(lambda v: v * v, (x,), name="squared")

    assert squared.value == 4.0 and squared.name == "squared"
