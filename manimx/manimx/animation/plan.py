"""Scene plans: a prepared scene as data, evaluated by the core.

    plan = ScenePlan(prepared)        # once, after Scene.prepare()
    plan.evaluate(2.5)                # any time, any order
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

Evaluating sets the evaluator's slots, not the Python signals: the
:class:`~.scene.PreparedScene` keeps its own state, and both can be
compared value by value (tests/test_plan.py).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .. import _core
from ..renderer.config import ViewState
from ..robjects.planned_frame import PlannedFrame
from .camera import Camera
from .camera_view import camera_views, current_value, view_cameras
from .clipping_group import ClippingGroup
from .contributions import (
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
from .curve3d import Curve3D
from .group import Group
from .kinds import (
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
from .light import DirectionalLight, PointLight, SpotLight
from .node import Node, Offset, PositionMixin, RotationMixin, ScalableMixin
from .particles import Particles
from .rates import linear, smooth
from .scene import PreparedScene
from .shape2d import Shape2D
from .shape3d import Shape3D
from .signal_tracks import Change, Entry, RateFunction, SignalTrack
from .signals import DerivedSignal, Signal

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
        self.graph = self._scene_graph()
        """The core's scene graph, or None when a node is of a class the
        core cannot draw (its scene is then drawn from snapshots)."""
        self._render_slots = sorted({self._slots[s.id] for root in scene.roots for node in root.family() for s in node.signals()})
        self._shader_slots = sorted({self.slot(node.surface_shader) for root in scene.roots for node in root.family() if isinstance(node, Shape3D)})
        """Slots of every surface shader: one in use makes a frame depend on time."""

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

    def frame_at(self, t: float, global_time: float | None = None) -> ViewState:
        """The view at compiled time ``t``: a :class:`PlannedFrame` and the
        scene's camera (what ``PreparedScene.frame_at`` returns as a tree).
        """
        self.evaluate(t, global_time)
        state = self.evaluator.state()
        value = lambda signal: state.value(self.slot(signal))  # noqa: E731
        return ViewState(
            PlannedFrame(self.graph, state, state.key(self._render_slots), any(state.value(slot) is not None for slot in self._shader_slots)),
            self.scene.camera.render_camera(value),
            camera_views(self.scene.roots, value),
        )

    def _scene_graph(self) -> Any:
        nodes: list[Node] = []
        index: dict[int, int] = {}
        # Cameras (animated ones are roots too) are never drawn.
        drawn_roots = [root for root in self.scene.roots if not isinstance(root, Camera)]
        for root in drawn_roots:
            for node in root.family():
                if id(node) not in index:
                    index[id(node)] = len(nodes)
                    nodes.append(node)
        described = []
        for node in nodes:
            kind = _node_kind(node)
            if kind is None:
                return None
            children: list[Node] = []
            if isinstance(node, Group):
                children = node.children
            elif isinstance(node, ClippingGroup):
                children = [node.mask, node.content]
            fields = {name: self.slot(signal) for name, signal in _node_fields(node)}
            described.append((kind, node.uid, fields, [index[id(child)] for child in children]))
        roots = [(index[id(root)], self.slot(root.present)) for root in drawn_roots]
        return _core.SceneGraph(described, roots)

    def _add_signal(self, signal: Signal) -> None:
        if signal.id in self._slots:
            return
        if isinstance(signal, DerivedSignal):
            for source in signal.inputs:
                self._add_signal(source)  # inputs first: dependency order
        self._slots[signal.id] = len(self._signals)
        self._signals.append(signal)


class PlannedTimeline:
    """A :class:`~.timing.Timeline` whose frames come from the core.

    ``frame_at(t)`` returns the same views as ``Timeline.frame_at(t)``,
    each with a :class:`PlannedFrame` instead of a tree. A clip whose
    scene has a node the core cannot draw falls back to its snapshot.
    Everything else (duration, audio, events) is the timeline's.
    """

    def __init__(self, timeline: Any) -> None:
        self.timeline = timeline
        self._plans: dict[int, ScenePlan] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.timeline, name)

    @property
    def complete(self) -> bool:
        """Whether every frame comes from the core: no clip falls back to
        snapshots.
        """
        self.timeline._ensure_resolved()
        return all(self.plan(clip.scene).graph is not None for clip in self.timeline.clips)

    def plan(self, scene: PreparedScene) -> ScenePlan:
        plan = self._plans.get(id(scene))
        if plan is None or plan.scene is not scene:
            plan = self._plans[id(scene)] = ScenePlan(scene)
        return plan

    def frame_at(self, t: float) -> tuple[ViewState, ...]:
        """Like ``Timeline.frame_at``: one view per active clip, in layer order."""
        timeline = self.timeline
        timeline._ensure_resolved()
        views = []
        for clip in sorted(timeline.clips, key=lambda clip: clip.layer):  # stable
            compiled = clip.compiled_time(t)
            if compiled is None:
                continue
            plan = self.plan(clip.scene)
            if plan.graph is None:
                state = clip.scene.state_at(compiled, global_time=t)
                views.append(ViewState(clip._group(state.children), clip.scene.camera.render_camera(), camera_views(clip.scene.roots, current_value)))
            else:
                views.append(plan.frame_at(compiled, global_time=t))
        return tuple(views)


_NODE_KINDS: tuple[tuple[type, str, str], ...] = (
    (Group, "group", "snapshot"),
    (ClippingGroup, "clipping_group", "snapshot"),
    (Particles, "particles", "_build_snapshot"),
    (Shape3D, "shape3d", "_build_snapshot"),
    (Shape2D, "shape2d", "_build_snapshot"),
    (Curve3D, "curve3d", "_build_snapshot"),
    (DirectionalLight, "directional_light", "_render_fields"),
    (PointLight, "point_light", "_render_fields"),
    (SpotLight, "spot_light", "_render_fields"),
)
"""Node classes the core draws: a subclass counts only while it keeps the
method that turns the class into render state."""

_FIELDS: dict[str, tuple[str, ...]] = {
    "group": (),
    "clipping_group": ("one_sided",),
    "shape2d": ("points", "fill_color", "fill_opacity", "stroke_color", "stroke_opacity", "stroke_width", "stroke_joint", "stroke_cap", "cast_shadows"),
    "curve3d": ("points", "stroke_color", "stroke_opacity", "stroke_width", "stroke_joint", "stroke_cap", "cast_shadows"),
    "shape3d": (
        "mesh", "base_color", "opacity", "metallic", "roughness", "emissive", "emissive_strength", "normal_scale",
        "occlusion_strength", "base_color_texture", "metallic_roughness_texture", "normal_texture",
        "occlusion_texture", "emissive_texture", "surface_shader", "cast_shadows",
    ),  # fmt: skip
    "directional_light": ("color", "intensity", "size", "cast_shadows"),
    "point_light": ("color", "intensity", "size", "cast_shadows", "range"),
    "spot_light": ("color", "intensity", "size", "cast_shadows", "range", "inner_cone_angle", "outer_cone_angle"),
}
_FIELDS["particles"] = (*_FIELDS["shape3d"], "positions", "rotations", "scales", "colors")


def _node_kind(node: Node) -> str | None:
    for cls, kind, method in _NODE_KINDS:
        if isinstance(node, cls):
            return kind if getattr(type(node), method) is getattr(cls, method) else None
    return None


def _node_fields(node: Node) -> Iterator[tuple[str, Signal]]:
    """``(field name, signal)`` of everything the core reads from ``node``."""
    yield "alpha", node.alpha
    yield "present", node.present
    if isinstance(node, PositionMixin):
        yield from ((f"position.{name}", axis) for name, axis in zip("xyz", node.position.axes, strict=True))
    if isinstance(node, RotationMixin):
        yield "rotation", node.rotation
    if isinstance(node, ScalableMixin):
        yield from ((f"scale.{name}", axis) for name, axis in zip("xyz", node.scale.axes, strict=True))
    for name in _FIELDS[_node_kind(node) or "group"]:
        yield name, getattr(node, name)


def planned_frames(source: Any) -> PlannedTimeline | PlannedScene | None:
    """The frames of a timeline or prepared scene through the core (an
    object with ``frame_at`` and ``complete``); None for anything else, and
    for a prepared scene the core cannot draw.
    """
    from .timing import Timeline

    if isinstance(source, Timeline):
        return PlannedTimeline(source)
    if isinstance(source, PreparedScene):
        plan = ScenePlan(source)
        return PlannedScene(plan) if plan.graph is not None else None
    return None


class PlannedScene:
    """A prepared scene's frames from the core, shaped like
    ``PreparedScene.frame_at`` (one view).
    """

    complete = True
    """Every frame comes from the core."""

    def __init__(self, plan: ScenePlan) -> None:
        self.plan = plan

    def frame_at(self, t: float) -> tuple[ViewState, ...]:
        return (self.plan.frame_at(t),)


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
    return ("python", function)


__all__ = ["PlannedScene", "PlannedTimeline", "ScenePlan", "planned_frames"]
