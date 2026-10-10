"""The tracks the page shows under each clip: the clip's nodes as a tree,
under each node the signals that animations change, and under each signal
the animations, placed on the global timeline.

    clip
    |- node (Group, Circle, ...)      children: its child nodes
       |- signal (position.x, ...)    only signals with an animation
          |- animation (FadeIn, animate.shift, ...)

Times are global: an animation's start and end in its scene are mapped
through the clip's segments (its cut, time map and start). An animation
that the cut skips entirely is left out.
"""

from __future__ import annotations

from typing import Any

from ..animation import ClippingGroup, Group, Node
from ..animation.animations import ChangeSignals, SetValue
from ..animation.signal_tracks import Placement, SignalTrack
from ..animation.timing import Clip


def clip_nodes(clip: Clip) -> list[dict[str, Any]]:
    """The node tree of ``clip``'s scene (camera first), as JSON values."""
    scene = clip.scene
    tracks = {track.signal.id: track for track in scene.tracks}
    roots = [scene.camera, *(root for root in scene.roots if root is not scene.camera)]
    # A node added on its own and also inside a group is shown in the group.
    nested = {id(node) for root in scene.roots for node in list(root.family())[1:]}
    return [_node(clip, root, tracks) for root in roots if id(root) not in nested]


def _node(clip: Clip, node: Node, tracks: dict[int, SignalTrack]) -> dict[str, Any]:
    signals = []
    for signal in node.signals():
        track = tracks.get(signal.id)
        animations = [] if track is None else [placed for entry in track.entries if (placed := _placed(clip, entry.placement)) is not None]
        if animations:
            signals.append({"name": _signal_name(node, signal.name), "animations": animations})
    return {
        "name": node.name,
        "type": type(node).__name__,
        "signals": signals,
        "children": [_node(clip, child, tracks) for child in _children(node)],
    }


def _children(node: Node) -> list[Node]:
    if isinstance(node, Group):
        return list(node.children)
    if isinstance(node, ClippingGroup):
        return [node.mask, node.content]
    return []


def _signal_name(node: Node, name: str) -> str:
    """``circle.position.x`` without the node's own name: ``position.x``."""
    prefix = f"{node.name}."
    return name[len(prefix) :] if name.startswith(prefix) else name


def _placed(clip: Clip, placement: Placement) -> dict[str, Any] | None:
    start = clip.global_time(placement.start)
    end = clip.global_time(placement.end)
    if start is None and end is None:
        return None
    start = end if start is None else start
    end = start if end is None else end
    return {"name": animation_label(placement.animation), "start": start, "end": end}


def animation_label(animation: Any) -> str:
    """How the page names an animation: its class, the methods of
    ``node.animate`` (``animate.shift.rotate``), or for an instant change
    ``add``, ``remove`` or ``set``."""
    if isinstance(animation, ChangeSignals):
        return ".".join(["animate", *animation.methods])
    if isinstance(animation, SetValue):
        change = animation.change
        if change.signal.name.endswith(".present"):
            return "add" if change.value else "remove"
        return "set"
    return type(animation).__name__


__all__ = ["animation_label", "clip_nodes"]
