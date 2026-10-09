"""Opt-in change tracking for render object fields.

Mark a dataclass field with :func:`versioned` instead of ``field`` and every
assignment to it bumps a per-object counter. Renderer caches put that counter
into their keys, so derived data (a fill triangulation, a GPU upload) is
rebuilt exactly when its inputs were replaced, and never when only an
unversioned field such as the transform changed.

In-place mutation (``shape.points[0] = ...``) is invisible to this scheme,
which is why versioned arrays are stored read-only.
"""

from __future__ import annotations

import dataclasses
from functools import cache
from typing import Any

_VERSIONED = "manimx.versioned"


def versioned(**field_kwargs: Any) -> Any:
    """``dataclasses.field`` that also opts the field into versioning."""
    metadata = {**field_kwargs.pop("metadata", {}), _VERSIONED: True}
    return dataclasses.field(metadata=metadata, **field_kwargs)


@cache
def versioned_fields(cls: type) -> frozenset[str]:
    """Names of all fields of dataclass ``cls`` declared with :func:`versioned`."""
    return frozenset(
        f.name for f in dataclasses.fields(cls) if f.metadata.get(_VERSIONED, False)
    )
