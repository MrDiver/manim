from __future__ import annotations

import itertools
from collections.abc import Hashable, Mapping
from dataclasses import KW_ONLY, InitVar, dataclass, field

from .transform import Quaternion, Transform
from .versioning import versioned_fields

__all__ = ["Quaternion", "RNode", "new_uid"]

# Runtime ids are never reused within a process, so renderer caches can key
# on them without ever confusing a new object with a deleted one.
_next_uid = itertools.count(1)


def new_uid() -> int:
    """A fresh process-unique uid.

    Objects get one automatically. The animation engine allocates one per
    scene object up front and passes it to every render object it builds
    for that scene object (``uid=``), so all snapshots of it share their
    identity, and the renderer's caches recognize it from frame to frame.
    """
    return next(_next_uid)


@dataclass(slots=True, eq=False)
class RNode:
    """Scene-graph node: identity, local 3D transform and visibility.

    Nodes compare by identity. Geometry and style live only on leaf objects
    such as :class:`RShape2D`; groups only arrange their children. Render
    objects carry no names: names and searching belong to the scene
    objects they are snapshots of (manimx.animation).

    Identity and change tracking, which the renderer's caches rely on:

    * :attr:`uid` is unique for the lifetime of the process. Pass ``uid=``
      to give a new object the identity of an earlier one: a snapshot of
      the same scene object at another time.
    * Fields declared with :func:`~.versioning.versioned` have a version
      token, see :meth:`version`. By default it counts assignments. Pass
      ``versions={"points": token, ...}`` to set the tokens explicitly,
      e.g. from the versions of the signals a snapshot was built from: then
      two snapshots built from unchanged signals have equal tokens and
      share cached data. Assigning a field afterwards still gives it a new,
      distinct token.
    """

    transform: Transform = field(default_factory=Transform)
    visible: bool = True
    """Whether flatten() draws this node and its subtree.

    Snapshots never set it: on the scene side, whether a node is drawn at
    all is decided before a render object exists. ``Node.present`` (set by
    ``Scene.add``/``remove``, by ``play`` when an animation starts, and by
    ``FadeOut``/``ShrinkToCenter`` when they end) selects the root nodes
    that ``PreparedScene.state_at()`` snapshots, and a node that should
    not be drawn snapshots to nothing (``snapshot()`` returns None, e.g.
    when scaled to zero). How strongly a node is drawn is its ``alpha``,
    which multiplies the opacities of its subtree; the renderer skips
    fills and strokes at opacity 0. A node hidden for a while is therefore
    faded out (and in again later), not switched off here. The field
    serves render trees built directly, e.g. in renderer tests.
    """
    _: KW_ONLY
    alpha: float = 1.0
    """Multiplies the opacity of this node and its whole subtree (fills,
    strokes and materials; a light's intensity). The snapshot of a
    ``Node`` carries its ``Node.alpha`` here. It is not versioned: like
    :attr:`transform`, the renderer applies it per object, so a fade
    changes one number instead of every color record.
    """
    uid: int = field(default_factory=new_uid)
    versions: InitVar[Mapping[str, Hashable] | None] = None
    _versions: dict[str, Hashable] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self, versions: Mapping[str, Hashable] | None) -> None:
        if not versions:
            return
        known = versioned_fields(type(self))
        unknown = sorted(set(versions) - known)
        if unknown:
            raise ValueError(f"{type(self).__name__} has no versioned fields {unknown}")
        self._versions.update(versions)

    def __setattr__(self, name: str, value: object) -> None:
        object.__setattr__(self, name, value)
        # _versions is declared before every subclass field, so it exists by
        # the time __init__ assigns a versioned field.
        if name in versioned_fields(type(self)):
            self._versions[name] = _next_token(self._versions.get(name, 0))

    def version(self, *fields: str) -> tuple[Hashable, ...]:
        """Version token of each named versioned field (a cache key part).

        Tokens are only compared for equality: equal tokens promise equal
        field values.
        """
        known = versioned_fields(type(self))
        for name in fields:
            if name not in known:
                raise ValueError(f"{type(self).__name__}.{name} is not versioned")
        return tuple(self._versions.get(name, 0) for name in fields)


def _next_token(token: Hashable) -> Hashable:
    """The token after an assignment: counters count up; explicit tokens
    get a suffix, so the result differs from every token seen before."""
    if isinstance(token, int):
        return token + 1
    if isinstance(token, tuple) and len(token) == 2 and token[0] == "assigned":
        return ("assigned", (token[1][0], token[1][1] + 1))
    return ("assigned", (token, 1))
