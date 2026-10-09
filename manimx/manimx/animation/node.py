"""Scene nodes: the authoring-side objects whose properties are signals.

A :class:`Node` is what scene code creates and animates. Every animatable
property is a signal, and :meth:`Node.snapshot` turns the current signal
values into render state (an :class:`~manimx.robjects.RNode`) for the
renderer. Snapshots reuse the node's ``uid`` and carry version tokens from
the signals, so renderer caches survive any number of snapshots (section
9 of the design doc).

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
from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence
from typing import Any

import numpy as np
import quaternion

from ..robjects import RNode, Transform, new_uid
from ..robjects.transform import quaternion_from_axis_angle
from ..utils.color import ManimColor, ParsableManimColor
from . import authoring
from .animations import AnimationBuilder
from .kinds import DISCRETE, FACTOR, ROTATION, SCALAR
from .signal_tracks import Change
from .signals import RootSignal, Signal, Vector3Signal, derive
from .snapshot import version_token

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
      opacity of the whole subtree.
    * ``present``: whether the node is in the scene; managed by the
      scene's ``add`` and ``remove``.

    Position, rotation and scale come from the mixins (see the module
    docs).
    """

    def __init__(self, name: str | None = None, *, tags: Iterable[str] = ()) -> None:
        self.uid = new_uid()
        self.name = name or f"{type(self).__name__.lower()}-{self.uid}"
        self.tags = _tag_set(tags)
        self.alpha = RootSignal(1.0, SCALAR, name=f"{self.name}.alpha")
        self.present = RootSignal(False, DISCRETE, name=f"{self.name}.present")
        self._last_snapshot: tuple[Hashable, RNode | None] | None = None

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

    @property
    def animate(self) -> AnimationBuilder:
        """Build an animation: ``self.play(square.animate.shift(RIGHT))``."""
        return AnimationBuilder(self)

    # -- changes ------------------------------------------------------------

    @animatable
    def set_alpha(self, alpha: float) -> ChangeList:
        return [(self.alpha, ABS, alpha)]

    # -- render state -------------------------------------------------------

    def snapshot(self, alphas: tuple[Signal, ...] = ()) -> RNode | None:
        """Render state for the current signal values (None if invisible).

        ``alphas`` are the ``alpha`` signals of the enclosing groups.

        While no signal of the node and no alpha above it changed, the
        previous snapshot is returned again: render nodes are immutable,
        and the renderer's caches then hit on the same object. Versions,
        not values, are compared (a revisited time builds once more)."""
        token = self._snapshot_token(alphas)
        last = self._last_snapshot
        if last is not None and last[0] == token:
            return last[1]
        snapshot = self._build_snapshot(alphas)
        self._last_snapshot = (token, snapshot)
        return snapshot

    def _snapshot_token(self, alphas: tuple[Signal, ...]) -> Hashable:
        """Changes whenever a snapshot could: a version of any signal of
        this node or of the alphas above it."""
        return (tuple(signal.version for signal in self.signals()), version_token(*alphas))

    def _build_snapshot(self, alphas: tuple[Signal, ...]) -> RNode | None:
        """Build the render state (see :meth:`snapshot`)."""
        raise NotImplementedError

    def _hidden(self) -> bool:
        """Nothing to draw (e.g. scaled to nothing)."""
        return False

    def _transform_parts(self) -> dict[str, Any]:
        """Keyword arguments of the snapshot's :class:`Transform`; each
        mixin adds its part."""
        return {}

    def _transform(self) -> Transform:
        return Transform(**self._transform_parts())


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

    def _transform_parts(self) -> dict[str, Any]:
        return {**super()._transform_parts(), "position": self.position.value}  # type: ignore[misc]

    def get_center(self) -> np.ndarray:
        """Position after everything recorded so far (while composing a scene)."""
        return np.array([authoring.current(axis) for axis in self.position.axes])

    @animatable
    def shift(self, vector: Sequence[float]) -> ChangeList:
        return [(axis, REL, float(component)) for axis, component in zip(self.position.axes, _vector(vector)) if component]

    @animatable
    def move_to(self, point: Sequence[float]) -> ChangeList:
        return [(axis, ABS, float(component)) for axis, component in zip(self.position.axes, _vector(point))]

    def follow(self, other: Node, offset: Sequence[float] = (0.0, 0.0, 0.0)) -> PositionMixin:
        """Keep this node at ``other``'s position plus ``offset`` (a
        derived signal per axis; the replacement for updaters)."""
        require(other, PositionMixin, f"{type(self).__name__}.follow()")
        for name, axis, delta in zip("xyz", other.position.axes, _vector(offset)):  # type: ignore[attr-defined]
            setattr(self.position, name, derive(lambda v, d=float(delta): v + d, axis, name=f"{self.name}.position.{name}"))  # type: ignore[attr-defined]
        return self


class RotationMixin:
    """A rotation: a quaternion signal."""

    _part = "rotation"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.rotation = RootSignal(quaternion.one, ROTATION, name=f"{self.name}.rotation")  # type: ignore[attr-defined]

    def signals(self) -> Iterator[Signal]:
        yield from super().signals()  # type: ignore[misc]
        yield self.rotation

    def _transform_parts(self) -> dict[str, Any]:
        return {**super()._transform_parts(), "rotation": self.rotation.value}  # type: ignore[misc]

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

    def _transform_parts(self) -> dict[str, Any]:
        return {**super()._transform_parts(), "scale": self.scale.value}  # type: ignore[misc]

    def _hidden(self) -> bool:
        return super()._hidden() or any(abs(axis.value) < 1e-9 for axis in self.scale.axes)  # type: ignore[misc]

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


__all__ = ["Node", "PositionMixin", "RotationMixin", "ScalableMixin", "animatable", "require"]
