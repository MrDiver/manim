"""Vector helpers ported from ``manim.utils.space_ops`` as manimx needs them."""

from __future__ import annotations

import numpy as np


def normalize(
    vect: np.ndarray | tuple[float], fall_back: np.ndarray | None = None
) -> np.ndarray:
    """Normalizes a vector to unit length while preserving its direction. If the vector
    has norm 0, a fallback vector is returned instead.

    Parameters
    ----------
    vect
        The vector to be normalized.
    fall_back
        The vector to be returned if ``vect`` has norm 0. If ``None``, a zero vector of
        the same length as ``vect`` is returned.
    """
    norm = np.linalg.norm(vect)
    if norm > 0:
        return np.array(vect) / norm
    else:
        return np.zeros(len(vect)) if fall_back is None else fall_back
