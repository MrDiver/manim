"""Growable record arrays that mirror GPU buffers.

The retained stores (retained.py for vector records, RetainedMeshes in
mesh_batch.py for mesh vertices and indices) keep their records in
arenas across renders, write only what changed, and hand the written
ranges to the renderer, which uploads just those.
"""

from __future__ import annotations

import numpy as np


class Arena:
    """A growable record array mirroring one GPU buffer, with the record
    ranges written since the last upload."""

    def __init__(self, dtype: np.dtype) -> None:
        self.dtype = dtype  # a subarray dtype (4 floats) must be kept as given
        self.data = np.zeros(256, dtype)
        self.used = 0
        self.garbage = 0
        self.dirty: list[tuple[int, int]] = []
        self.reallocated = True

    def allocate(self, count: int) -> int:
        start = self.used
        if start + count > len(self.data):
            grown = np.zeros(max(2 * len(self.data), start + count), self.dtype)
            grown[:start] = self.data[:start]
            self.data = grown
            self.reallocated = True
        self.used += count
        return start

    def write(self, start: int, records: np.ndarray) -> None:
        self.data[start : start + len(records)] = records
        self.dirty.append((start, start + len(records)))

    def touch(self, start: int, stop: int) -> None:
        self.dirty.append((start, stop))

    def view(self) -> np.ndarray:
        return self.data[: self.used]

    def ranges(self) -> list[tuple[int, int]] | None:
        """Merged dirty ranges since the last call, or None for "upload
        everything" (after a reallocation)."""
        if self.reallocated:
            self.reallocated = False
            self.dirty = []
            return None
        merged: list[tuple[int, int]] = []
        for start, stop in sorted(self.dirty):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
            else:
                merged.append((start, stop))
        self.dirty = []
        return merged

    def place(self, records: np.ndarray, old_first: int, old_count: int) -> int:
        """Write ``records`` over the old range if they fit exactly, else into
        a new range (the old one becomes garbage); returns the first index.
        """
        if old_count == len(records) and old_count > 0:
            self.write(old_first, records)
            return old_first
        self.garbage += old_count
        first = self.allocate(len(records))
        self.write(first, records)
        return first
