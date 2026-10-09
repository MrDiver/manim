"""Find snapshots by the name of the Node they were built from.

Render objects carry no names; names belong to the Nodes. A snapshot
keeps its Node's uid, so the Node tree maps uids back to names.
"""

from __future__ import annotations

from collections.abc import Iterable

from manimx.animation import Node, Timeline
from manimx.robjects import RNode


def by_name(snapshots: Iterable[RNode], roots: Iterable[Node]) -> dict[str, RNode]:
    """``snapshots`` keyed by the names of the Nodes in ``roots``' families."""
    names = {node.uid: node.name for root in roots for node in root.family()}
    return {names[snapshot.uid]: snapshot for snapshot in snapshots}


def timeline_roots(timeline: Timeline) -> list[Node]:
    """The root Nodes of every clip's scene."""
    return [root for clip in timeline.clips for root in clip.scene.roots]


def clip_names(groups: Iterable[RNode], timeline: Timeline) -> list[str]:
    """The names of the clips whose render groups are ``groups``."""
    names = {clip.uid: clip.name for clip in timeline.clips}
    return [names[group.uid] for group in groups]
