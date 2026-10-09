"""Helpers the examples share for building node trees."""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import quaternion

from ..animation import Node, PositionMixin, RotationMixin, ScalableMixin
from ..robjects import Transform

N = TypeVar("N", bound=Node)


def placed(node: N, transform: Transform) -> N:
    """``node`` with its position, rotation and scale signals set to
    ``transform`` (outside a scene, so no animation is recorded).

    A part the node does not have must be the identity in ``transform``.
    """
    parts = (
        (PositionMixin, "position", transform.position, not np.any(transform.position)),
        (RotationMixin, "rotation", transform.rotation, transform.rotation == quaternion.one),
        (ScalableMixin, "scale", transform.scale, transform.scale == (1.0, 1.0, 1.0)),
    )
    for mixin, name, value, identity in parts:
        if isinstance(node, mixin):
            getattr(node, name).set(value)
        elif not identity:
            raise ValueError(f"{type(node).__name__} has no {name}, but the transform sets one")
    return node
