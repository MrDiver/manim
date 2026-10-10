"""PlannedFrame: what a scene shows at one time, as the core packs it.

The other form of render state next to the ``RNode`` tree (see CORE.md):
instead of render nodes it carries the core's scene graph and the values
of every signal at that time (``manimx.animation.plan.ScenePlan`` makes
them). The render chain hands it to the core's packer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True, eq=False)
class PlannedFrame:
    """A frame of the core; it takes the place of the tree in a ViewState."""

    graph: Any
    """``manimx._core.SceneGraph``."""
    state: Any
    """``manimx._core.FrameState``."""
    key: bytes
    """Equal keys (with the same graph) promise equal pictures: the
    versions of every slot a node reads."""


__all__ = ["PlannedFrame"]
