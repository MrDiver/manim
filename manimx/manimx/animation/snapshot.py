"""Helpers for building render state (manimx.objects) from signals.

Section 9 of docs/animation_engine_design.md: every snapshot of a scene
object reuses the object's ``uid``, and its versioned fields carry tokens
derived from the signals they were computed from. The renderer then keeps
cached data (fill triangulations, packed vertices) for as long as those
signals do not change, no matter how many snapshots are built.

Tokens are change counters, not content hashes. Evaluating t = 5, then
t = 2, then t = 5 again yields a different token for an animated signal the
second time, although the value is the same: that costs one recomputation
and is never wrong. Hashing every value instead (for example large point
arrays) would cost time on every frame for the rare case of revisiting a
time.
"""

from __future__ import annotations

from collections.abc import Hashable

from .signals import Signal, Vector3Signal


def version_token(*signals: Signal | Vector3Signal) -> Hashable:
    """A token that changes whenever any of ``signals`` changes value.

    It combines each signal's id and version, so tokens of different signal
    sets never collide, and reading it brings derived signals up to date.
    """
    parts: list[tuple[int, int]] = []
    for signal in signals:
        axes = signal.axes if isinstance(signal, Vector3Signal) else (signal,)
        parts.extend((axis.id, axis.version) for axis in axes)
    return tuple(parts)
