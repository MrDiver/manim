"""The authoring context: what happens to a change while ``construct()`` runs.

Section 4.3 and 5 of docs/animation_engine_design.md. Scene objects do not
know whether they are being set up, animated, or evaluated; they route
every change and every authoring-time read through this module:

* **No scene is being compiled** (setting objects up, or after
  ``prepare()``): a change sets the signal's value directly.
* **A scene is being compiled and the change concerns an object that is
  already on the timeline:** the change becomes an instantaneous animation
  (:class:`SetValue`) placed at the current cursor time. Setting the signal
  instead would rewrite the object's past: ``square.shift(RIGHT)`` between
  two plays must not move the square at t = 0.
* **Authoring-time reads** (``get_center()``) return the value after
  everything recorded so far, like Manim, so ``move_to(other.get_center())``
  after an animation refers to where ``other`` ended up.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from .signals import RootSignal, Signal
from .tracks import Change

if TYPE_CHECKING:
    from .scene import SceneCompiler

_active: ContextVar[SceneCompiler | None] = ContextVar("manimx_scene_compiler", default=None)


def active_compiler() -> SceneCompiler | None:
    """The compiler of the scene whose ``construct()`` is running, if any."""
    return _active.get()


def change(signal: Signal, mode: Change, value: Any) -> None:
    """Apply a change from authoring code (see the module docs)."""
    if not isinstance(signal, RootSignal):
        raise TypeError(f"{signal.name} is derived from other signals and cannot be changed directly")
    compiler = active_compiler()
    if compiler is not None and compiler.on_timeline(signal):
        compiler.record_instant(signal, mode, value)
        return
    kind = signal.kind
    signal.set(kind.combine(signal.value, value) if mode is Change.RELATIVE else value)


def current(signal: Signal) -> Any:
    """The value of ``signal`` as authoring code should see it right now."""
    compiler = active_compiler()
    if compiler is not None:
        return compiler.current_value(signal)
    return signal.value
