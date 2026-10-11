"""Scene nodes: the authoring-side objects whose properties are signals.

A :class:`Node` is what scene code creates and animates. Every animatable
property is a signal. A node says once how it is drawn: its
:attr:`~Node.render_kind` (which render object it becomes) and
:meth:`~Node.render_fields` (which signal feeds which field). From that
the core builds the render object (an :class:`~manimx.robjects.RNode`) of
any time (manimx/animation/plan.py); :meth:`Node.snapshot` builds the one
of the current values. Snapshots carry the node's ``uid``, and their
values keep their versions while the signals do not change, so renderer
caches survive any number of snapshots.

The scene tree is parallel to the render tree: every node class
snapshots to exactly one render class. Only :class:`Group` and
:class:`ClippingGroup` have children; everything else is a leaf.

The transform is split by meaning. :class:`PositionMixin`,
:class:`RotationMixin` and :class:`ScalableMixin` each add one part (its
signal and the changes that use it), and a node class takes the parts
that mean something for it: a directional light has a direction but no
position, a point light a position but no direction, and no light has a
scale. A missing part is the identity in the snapshot's transform.

Change methods (``shift``, ``move_to``, ``set_fill``, ...) are defined once
with :func:`animatable`, as a function returning ``(signal, change,
value)`` triples. Called directly, the changes go through the authoring
context (authoring.py); called on ``node.animate``, they are collected
into an animation. Both paths share one definition.
"""

from __future__ import annotations

import difflib
import functools
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import quaternion

from ...robjects import RNode, new_uid
from ...robjects.transform import quaternion_from_axis_angle
from ...utils.color import ManimColor, ParsableManimColor
from .. import authoring
from ..animations import AnimationBuilder
from ..signals.kinds import DISCRETE, FACTOR, ROTATION, SCALAR
from ..signals.paths import PointSignal
from ..signals.signal import RootSignal, Signal, Vector3Signal, derive
from ..signals.signal_tracks import Change

if TYPE_CHECKING:
    from ..scene import PreparedScene

ChangeList = list[tuple[RootSignal, Change, Any]]

REL, ABS = Change.RELATIVE, Change.ABSOLUTE


def animatable(changes: Callable[..., ChangeList]) -> Callable[..., Any]:
    """Turn a change definition into a method usable directly and via
    ``.animate`` (see the module docs)."""

    @functools.wraps(changes)
    def method(self: Node, *args: Any, **kwargs: Any) -> Node:
        for signal, mode, value in changes(self, *args, **kwargs):
            authoring.change(signal, mode, value)
        return self

    method._signal_changes = changes  # type: ignore[attr-defined]
    return method


class Node:
    """Something in a scene: identity, names, an opacity multiplier and
    presence.

    * ``uid``: the identity its snapshots carry (render objects have no
      names; see the render contract).
    * ``name`` and ``tags``: for finding nodes while composing a scene
      (``group.find(name)``, ``group.find_all(tags=...)``), like an HTML
      id and classes. A name is meant to identify one node but is not
      checked for uniqueness: imported models often repeat names.
    * ``alpha``: an opacity multiplier for fades; it multiplies the
      opacity of the whole subtree. Snapshots carry it as
      ``RNode.alpha``, and the renderer multiplies it down the tree, so
      a fade never rebuilds the fills and strokes below.
    * ``present``: whether the node is in the scene; managed by the
      scene's ``add`` and ``remove``.

    Position, rotation and scale come from the mixins (see the module
    docs).
    """

    render_kind: str | None = None
    """Which render object the node becomes: ``"group"``,
    ``"clipping_group"``, ``"shape2d"``, ``"curve3d"``, ``"shape3d"``,
    ``"particles"``, ``"directional_light"``, ``"point_light"`` or
    ``"spot_light"`` (core/src/evaluation/scene_graph.rs builds them).
    None for nodes that are never drawn (a Camera, a Value): they snapshot
    to None."""

    def __init__(self, name: str | None = None, *, tags: Iterable[str] = ()) -> None:
        self.uid = new_uid()
        self.name = name or f"{type(self).__name__.lower()}-{self.uid}"
        self.tags = _tag_set(tags)
        self.alpha = RootSignal(1.0, SCALAR, name=f"{self.name}.alpha")
        self.present = RootSignal(False, DISCRETE, name=f"{self.name}.present")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"

    def __getattr__(self, name: str) -> Any:
        # Only called for missing attributes: explain the ones a mixin
        # would provide, instead of a bare AttributeError.
        mixin = _MIXIN_OF_ATTRIBUTE.get(name)
        if mixin is not None:
            raise AttributeError(f"{type(self).__name__} has no {mixin._part}, so it has no {name!r} ({mixin.__name__} provides it)")
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    # -- structure ----------------------------------------------------------

    def family(self) -> Iterator[Node]:
        """This node and all its descendants (only groups have any)."""
        yield self

    def signals(self) -> Iterator[Signal]:
        """Every signal of this node (not of its children)."""
        yield self.alpha
        yield self.present

    def after_prepare(self, scene: PreparedScene) -> bool:
        """Called once with the prepared scene this node is in, before it is
        shown. A node whose signals depend on the whole timeline (a
        TracedPath samples where a point was) sets them here and returns
        True; the scene's evaluator is then made again with the new values.
        """
        return False

    @property
    def animate(self) -> AnimationBuilder:
        """Build an animation: ``self.play(square.animate.shift(RIGHT))``."""
        return AnimationBuilder(self)

    # -- changes ------------------------------------------------------------

    @animatable
    def set_alpha(self, alpha: float) -> ChangeList:
        return [(self.alpha, ABS, alpha)]

    # -- render state -------------------------------------------------------

    def render_fields(self) -> Iterator[tuple[str, Signal]]:
        """``(field, signal)`` of everything the render object of this node
        is made of: ``alpha`` and ``present`` here, the transform parts
        from the mixins, the rest from each node class."""
        yield "alpha", self.alpha
        yield "present", self.present

    def render_children(self) -> list[Node]:
        """The nodes whose render objects are this one's children."""
        return []

    def snapshot(self) -> RNode | None:
        """The render object of this node and everything below it for the
        current signal values, built by the core like every frame's; None
        for a node that is not drawn (never drawn, or scaled to nothing).

        ``alpha`` becomes its ``RNode.alpha``; the alphas of the
        enclosing groups are multiplied in by whoever reads the tree.
        """
        from ..plan import snapshot

        return snapshot(self)


class PositionMixin:
    """A position: one signal per axis, so animations of different axes
    never interfere."""

    _part = "position"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.position = Vector3Signal(name=f"{self.name}.position")  # type: ignore[attr-defined]

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()  # type: ignore[misc]
        yield from self.position.axes

    def render_fields(self) -> Iterator[tuple[str, Signal]]:
        yield from super().render_fields()  # type: ignore[misc]
        yield from ((f"position.{name}", axis) for name, axis in zip("xyz", self.position.axes, strict=True))


    def get_center(self) -> np.ndarray:
        """Position after everything recorded so far (while composing a scene)."""
        return np.array([authoring.current(axis) for axis in self.position.axes])

    @animatable
    def shift(self, vector: Sequence[float]) -> ChangeList:
        return [(axis, REL, float(component)) for axis, component in zip(self.position.axes, _vector(vector)) if component]

    @animatable
    def move_to(self, point: Sequence[float]) -> ChangeList:
        return [(axis, ABS, float(component)) for axis, component in zip(self.position.axes, _vector(point))]

    def follow(self, other: Node | PointSignal, offset: Sequence[float] = (0.0, 0.0, 0.0)) -> PositionMixin:
        """Keep this node at ``other``'s position plus ``offset`` (a
        derived signal per axis; the replacement for updaters). ``other``
        is a node or a point that follows signals, such as
        ``shape.point_at(0.5)``."""
        follow_point(self.position, other, offset, f"{self.name}.position", f"{type(self).__name__}.follow()")  # type: ignore[attr-defined]
        return self


def follow_point(
    point: Vector3Signal,
    other: Node | PointSignal,
    offset: Sequence[float] = (0.0, 0.0, 0.0),
    name: str = "",
    caller: str = "follow_point()",
) -> None:
    """Make the axes of ``point`` (a position, a Line's end) follow
    ``other``'s position or a PointSignal, plus ``offset``: each axis
    becomes a signal derived from the other's (see PositionMixin.follow).
    Signals already derived from an axis (a Line's points from its end)
    are rebound to the new one."""
    if isinstance(other, PointSignal):
        axes = other.axes
    else:
        require(other, PositionMixin, caller)
        axes = other.position.axes  # type: ignore[attr-defined]
    for axis_name, axis, delta in zip("xyz", axes, _vector(offset), strict=True):
        old = getattr(point, axis_name)
        new = derive(Offset(float(delta)), axis, name=f"{name}.{axis_name}")
        setattr(point, axis_name, new)
        for dependent in list(old._dependents):
            dependent.rebind(dependent.function, tuple(new if source is old else source for source in dependent.inputs))


class Offset:
    """``value + offset``: the function of a :meth:`PositionMixin.follow`
    axis. A class instead of a lambda, so the core (manimx._core)
    recognizes it and adds the offset without calling Python."""

    __slots__ = ("offset",)

    def __init__(self, offset: float) -> None:
        self.offset = offset

    def __call__(self, value: float) -> float:
        return value + self.offset


class RotationMixin:
    """A rotation: a quaternion signal."""

    _part = "rotation"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.rotation = RootSignal(quaternion.one, ROTATION, name=f"{self.name}.rotation")  # type: ignore[attr-defined]

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()  # type: ignore[misc]
        yield self.rotation

    def render_fields(self) -> Iterator[tuple[str, Signal]]:
        yield from super().render_fields()  # type: ignore[misc]
        yield "rotation", self.rotation


    @animatable
    def rotate(self, angle: float, axis: Sequence[float] = (0.0, 0.0, 1.0)) -> ChangeList:
        """Rotate by ``angle``; animated, it takes the shortest way (use the
        Rotate animation to sweep full turns)."""
        return [(self.rotation, REL, quaternion_from_axis_angle(axis, angle))]


class ScalableMixin:
    """A scale: one factor signal per axis. A node scaled to nothing on any
    axis is not drawn (a zero scale would also leave a shape's plane
    without a normal)."""

    _part = "scale"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.scale = Vector3Signal((1.0, 1.0, 1.0), name=f"{self.name}.scale", kind=FACTOR)  # type: ignore[attr-defined]

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()  # type: ignore[misc]
        yield from self.scale.axes

    def render_fields(self) -> Iterator[tuple[str, Signal]]:
        yield from super().render_fields()  # type: ignore[misc]
        yield from ((f"scale.{name}", axis) for name, axis in zip("xyz", self.scale.axes, strict=True))


    @animatable
    def scale_by(self, factor: float | Sequence[float]) -> ChangeList:
        factors = np.broadcast_to(np.asarray(factor, dtype=np.float64), (3,))
        return [(axis, REL, float(f)) for axis, f in zip(self.scale.axes, factors) if f != 1.0]

    @animatable
    def set_scale(self, scale: float | Sequence[float]) -> ChangeList:
        """Set the scale (absolute; ``scale_by`` multiplies instead)."""
        values = np.broadcast_to(np.asarray(scale, dtype=np.float64), (3,))
        return [(axis, ABS, float(v)) for axis, v in zip(self.scale.axes, values)]


_MIXIN_OF_ATTRIBUTE: dict[str, type] = {
    name: mixin
    for mixin in (PositionMixin, RotationMixin, ScalableMixin)
    for name in (mixin._part, *(n for n, v in vars(mixin).items() if not n.startswith("_") and callable(v)))
}
"""Attributes a mixin provides, for the error a node without it raises."""


def require(node: Node, mixin: type, action: str) -> None:
    """Raise a TypeError naming ``action`` unless ``node`` has ``mixin``'s part."""
    if not isinstance(node, mixin):
        part = mixin._part  # type: ignore[attr-defined]
        raise TypeError(f"{action} needs a {part}, but {type(node).__name__} {node.name!r} has none ({mixin.__name__} provides it)")


def find(root: Node, name: str) -> Node:
    """The one node named ``name`` below ``root`` (see Group.find)."""
    matches = [node for node in _below(root) if node.name == name]
    if len(matches) == 1:
        return matches[0]
    where = f"{type(root).__name__} {root.name!r}"
    if matches:
        raise LookupError(f"{where} has {len(matches)} nodes named {name!r}; use find_all(name=...) to get all of them")
    close = difflib.get_close_matches(name, sorted({node.name for node in _below(root)}), n=3)
    hint = f"; did you mean {', '.join(repr(c) for c in close)}?" if close else ""
    raise LookupError(f"{where} has no node named {name!r}{hint}")


def find_all(root: Node, name: str | None = None, tags: Iterable[str] = ()) -> list[Node]:
    """Every node below ``root`` named ``name`` (if given) that has all of
    ``tags``, in tree order (see Group.find_all)."""
    wanted = _tag_set(tags)
    return [node for node in _below(root) if (name is None or node.name == name) and wanted <= node.tags]


def _below(root: Node) -> Iterator[Node]:
    """``root``'s family without ``root`` itself."""
    family = root.family()
    next(family)
    return family


def _tag_set(tags: Iterable[str]) -> frozenset[str]:
    """Tags as a set; a single string is one tag, not its characters."""
    values = (tags,) if isinstance(tags, str) else tuple(tags)
    for tag in values:
        if not isinstance(tag, str):
            raise TypeError(f"tags must be strings, got {tag!r}")
    return frozenset(values)


def _rgb(color: ParsableManimColor) -> tuple[float, float, float]:
    r, g, b = ManimColor.parse(color).to_rgb()
    return (float(r), float(g), float(b))


def _vector(value: Sequence[float]) -> tuple[float, float, float]:
    array = np.zeros(3)
    given = np.asarray(value, dtype=np.float64).reshape(-1)
    array[: len(given)] = given
    return (float(array[0]), float(array[1]), float(array[2]))


__all__ = ["Node", "PositionMixin", "RotationMixin", "ScalableMixin", "animatable", "follow_point", "require"]
