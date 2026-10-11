"""RNode: the base of render state, what a scene node shows at one time.

``PreparedScene.state_at(t)`` snapshots every scene node
(manimx.animation) into a tree of render objects: ``RGroup``,
``RClippingGroup``, ``RShape2D``, ``RCurve3D``, ``RShape3D``,
``RParticles`` and the lights. Frames are not rendered from these trees
(the core reads the signals itself, see manimx/animation/plan.py);
the trees are for reading a scene's state in tests and tools, e.g. the
web UI's labels.
"""

from __future__ import annotations

import itertools
from dataclasses import KW_ONLY, dataclass, field

from .transform import Quaternion, Transform

__all__ = ["Quaternion", "RNode", "new_uid"]

# Ids are counted, never reused within a process, so a new object is never
# mistaken for a deleted one.
_next_uid = itertools.count(1)


def new_uid() -> int:
    """A fresh process-unique uid.

    Scene nodes get one when they are made and pass it to every render
    object built for them (``uid=``), so all snapshots of a node share its
    identity.
    """
    return next(_next_uid)


@dataclass(slots=True, eq=False)
class RNode:
    """Scene-graph node: identity, local 3D transform and alpha.

    Nodes compare by identity. Geometry and style live only on leaf objects
    such as :class:`RShape2D`; groups only arrange their children. Render
    objects carry no names: names and searching belong to the scene
    objects they are snapshots of (manimx.animation).

    A node that should not be drawn snapshots to nothing
    (``Node.snapshot()`` returns None, e.g. when scaled to zero), and only
    present nodes are snapshot at all (``Node.present``).
    """

    transform: Transform = field(default_factory=Transform)
    _: KW_ONLY
    alpha: float = 1.0
    """Multiplies the opacity of this node and its whole subtree (fills,
    strokes and materials; a light's intensity). The snapshot of a
    ``Node`` carries its ``Node.alpha`` here.
    """
    uid: int = field(default_factory=new_uid)
    """Unique for the lifetime of the process; all snapshots of one scene
    node share it."""
