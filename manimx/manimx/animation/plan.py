"""Scene plans: a prepared scene as data, evaluated by the core.

    plan = ScenePlan(prepared)        # once (prepared.plan makes it)
    plan.state_at(2.5)                # the render object tree at 2.5
    plan.value(square.position.x)     # what the signal has at 2.5

A :class:`ScenePlan` turns a :class:`~.scene.PreparedScene` into the
tables of the core's evaluator (``manimx._core.Evaluator``, see CORE.md):

* **signals:** every signal the scene can read, each with a slot: the
  time signals, the signals of every track, of every node in the scene
  and of the camera, and every input of a derived signal. A derived
  signal comes after its inputs, so the evaluator computes them in slot
  order.
* **tracks:** per animated root signal its base value and its entries.
  An entry's contribution is data when its animation describes it
  (:meth:`~.animations.Animation.planned_contribution`), else a call back
  to ``Animation.contribution()``. Rates are data for ``linear`` and
  ``smooth``, else called back.
* **derived signals:** offsets of :meth:`~.node.PositionMixin.follow` are
  data; every other function is called back with the input values.
* **scene graph:** every drawn node with its ``render_kind``, the slots of
  its ``render_fields`` and its ``render_children``. From the values at
  a time the core builds the render object tree (``manimx.robjects``),
  the one interface between a scene and whatever renders it.

Evaluating sets the evaluator's slots, not the Python signals: the
:class:`~.scene.PreparedScene` keeps its own state, and both can be
compared value by value (tests/test_plan.py).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .. import _core
from ..renderer.config import ViewState
from ..robjects import RGroup, RNode
from .nodes.camera_view import camera_view_nodes, camera_views, view_cameras
from .nodes.node import Node, Offset
from .scene import PreparedScene
from .signals.contributions import (
    Between,
    By,
    Contribution,
    Keyframes,
    Partial,
    Power,
    Ramp,
    To,
    Turn,
)
from .signals.kinds import (
    ArrayKind,
    ColorKind,
    ColorsKind,
    DiscreteKind,
    FactorKind,
    PathKind,
    RotationKind,
    ScalarKind,
    ScalarsKind,
    SignalKind,
)
from .signals.paths import DrawnPart
from .signals.rates import linear, smooth
from .signals.signal import DerivedSignal, Signal
from .signals.signal_tracks import Change, Entry, RateFunction, SignalTrack

_KIND_CODES: dict[type, str] = {
    ScalarKind: "scalar",
    FactorKind: "factor",
    RotationKind: "rotation",
    ColorKind: "color",
    ColorsKind: "colors",
    ScalarsKind: "scalars",
    PathKind: "path",
    ArrayKind: "array",
    DiscreteKind: "discrete",
}
"""Kinds the core implements, by exact type (a subclass may change any
operation, so it is called back like any other kind)."""


class ScenePlan:
    """A prepared scene in the core's evaluator (see the module docs)."""

    def __init__(self, scene: PreparedScene) -> None:
        self.scene = scene
        self._slots: dict[int, int] = {}
        self._signals: list[Signal] = []
        for signal in _readable_signals(scene):
            self._add_signal(signal)
        self.evaluator = _core.Evaluator(
            [_kind_code(signal.kind) for signal in self._signals],
            [signal.value for signal in self._signals],
            self.slot(scene.time.scene_time),
            self.slot(scene.time.global_time),
        )
        for track in scene.tracks:
            self.evaluator.add_track(self.slot(track.signal), track.base, [_entry(entry, track) for entry in track.entries])
        for signal in self._signals:
            if isinstance(signal, DerivedSignal):
                self.evaluator.add_derived(self.slot(signal), [self.slot(i) for i in signal.inputs], _derivation(signal))
        self.graph = scene_graph(scene.roots, self.slot, lambda root: self.slot(root.present))
        """The core's scene graph."""
        self._camera_views = camera_view_nodes(scene.roots)

    @property
    def signals(self) -> list[Signal]:
        """Every signal in the plan, in slot order."""
        return list(self._signals)

    def slot(self, signal: Signal) -> int:
        return self._slots[signal.id]

    def evaluate(self, t: float, global_time: float | None = None) -> None:
        """Set every slot to its value at compiled time ``t``, like
        :meth:`PreparedScene.evaluate` sets the signals.
        """
        self.evaluator.evaluate(t, self.scene.playing_time_at(t), t if global_time is None else global_time)

    def value(self, signal: Signal) -> Any:
        """The value of ``signal`` at the last evaluated time."""
        return self.evaluator.value(self.slot(signal))

    def version(self, signal: Signal) -> int:
        """Increases exactly when the value of ``signal`` changes."""
        return self.evaluator.version(self.slot(signal))

    def state_at(self, t: float, global_time: float | None = None, uid: int | None = None) -> RGroup:
        """The render object tree at compiled time ``t``: an RGroup (with
        uid ``uid``, by default the scene's) holding the present roots in
        the order they were added.
        """
        self.evaluate(t, global_time)
        return self.graph.snapshot(self.evaluator.state(), self.scene.uid if uid is None else uid)

    def frame_at(self, t: float, global_time: float | None = None, uid: int | None = None) -> ViewState:
        """The view at compiled time ``t``: :meth:`state_at` seen through
        the scene's camera, with the scene's camera views.
        """
        root = self.state_at(t, global_time, uid)
        evaluator = self.evaluator
        value = lambda signal: evaluator.value(self.slot(signal))  # noqa: E731
        return ViewState(root, self.scene.camera.render_camera(value), camera_views(self._camera_views, value))

    def _add_signal(self, signal: Signal) -> None:
        if signal.id in self._slots:
            return
        if isinstance(signal, DerivedSignal):
            for source in signal.inputs:
                self._add_signal(source)  # inputs first: dependency order
        self._slots[signal.id] = len(self._signals)
        self._signals.append(signal)


def scene_graph(roots: Sequence[Node], slot: Callable[[Signal], int], present: Callable[[Node], int | None]) -> Any:
    """The core's ``SceneGraph`` of the drawn nodes under ``roots``: each
    node's ``render_kind``, uid, the slot (``slot(signal)``) of each of
    its ``render_fields`` and its drawn ``render_children``; each root with
    the slot of the signal that says whether it is shown (None: always).
    Nodes that are never drawn (cameras, values) are left out.
    """
    nodes: list[Node] = []
    index: dict[int, int] = {}  # node uid -> position in nodes

    def add(node: Node) -> None:
        if node.uid in index:
            return
        index[node.uid] = len(nodes)
        nodes.append(node)
        for child in node.render_children():
            if child.render_kind is not None:
                add(child)

    drawn_roots = [root for root in roots if root.render_kind is not None]
    for root in drawn_roots:
        add(root)
    described = []
    for node in nodes:
        children = node.render_children()
        if node.render_kind == "clipping_group" and any(child.render_kind is None for child in children):
            raise TypeError(f"{node!r}: the mask and the content of a ClippingGroup must be nodes that are drawn")
        fields = {name: slot(signal) for name, signal in node.render_fields()}
        described.append((node.render_kind, node.uid, fields, [index[child.uid] for child in children if child.uid in index]))
    return _core.SceneGraph(described, [(index[root.uid], present(root)) for root in drawn_roots])


def snapshot(node: Node) -> RNode | None:
    """``node.snapshot()``: the render object of ``node`` (and everything
    below it) for the current values of its signals, built by the core
    like every frame's; None if it is not drawn."""
    if node.render_kind is None:
        return None
    signals: list[Signal] = []
    slots: dict[int, int] = {}

    def slot(signal: Signal) -> int:
        if signal.id not in slots:
            slots[signal.id] = len(signals)
            signals.append(signal)
        return slots[signal.id]

    graph = scene_graph([node], slot, lambda root: None)
    state = _core.FrameState([_kind_code(signal.kind) for signal in signals], [signal.value for signal in signals])
    children = graph.snapshot(state, 0).children
    return children[0] if children else None


def _readable_signals(scene: PreparedScene) -> list[Signal]:
    """Every signal evaluation sets or a snapshot reads."""
    signals: list[Signal] = [scene.time.scene_time, scene.time.global_time]
    signals += [track.signal for track in scene.tracks]
    for root in scene.roots:
        for node in root.family():
            signals += node.signals()
    signals += scene.camera.signals()
    for camera in view_cameras(scene.roots):
        signals += camera.signals()
    return signals


def _kind_code(kind: SignalKind) -> tuple[str, SignalKind]:
    return (_KIND_CODES.get(type(kind), "python"), kind)


def _entry(entry: Entry, track: SignalTrack) -> tuple:
    placement = entry.placement
    planned = placement.animation.planned_contribution(track.signal)  # type: ignore[attr-defined]
    contribution = _contribution(planned) if planned is not None else ("python", placement.animation, track.signal)
    return (
        placement.start,
        placement.duration,
        _rate(placement.rate),
        entry.mode is Change.RELATIVE,
        entry.start_value,
        contribution,
    )


def _rate(rate: RateFunction) -> tuple:
    if rate is linear:
        return ("linear",)
    if rate is smooth:
        return ("smooth", 10.0)
    return ("python", rate)


def _contribution(planned: Contribution) -> tuple:
    if isinstance(planned, To):
        return ("to", planned.value)
    if isinstance(planned, By):
        return ("by", planned.change)
    if isinstance(planned, Ramp):
        return ("ramp", planned.factor, planned.remaining)
    if isinstance(planned, Power):
        return ("power", planned.base)
    if isinstance(planned, Turn):
        return ("turn", planned.axis, planned.angle)
    if isinstance(planned, Keyframes):
        return ("keyframes", planned.run_time, list(planned.times), list(planned.values))
    if isinstance(planned, Between):
        return ("between", planned.start, planned.end, planned.lower, planned.upper)
    if isinstance(planned, Partial):
        return ("partial", planned.points, planned.lower, planned.upper, planned.reverse)
    raise TypeError(f"unknown contribution {planned!r}")


def _derivation(signal: DerivedSignal) -> tuple:
    function = signal.function
    if isinstance(function, Offset):
        return ("offset", function.offset)
    if isinstance(function, DrawnPart):
        return ("drawn_part",)
    return ("python", function)


__all__ = ["ScenePlan", "scene_graph", "snapshot"]
