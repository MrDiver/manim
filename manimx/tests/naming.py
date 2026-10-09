"""Find snapshots by the name of the Mobject they were built from.

Render objects carry no names; names belong to the Mobjects. A snapshot
keeps its Mobject's uid, so the Mobject tree maps uids back to names.
"""

from __future__ import annotations

from collections.abc import Iterable

from manimx.animation import Mobject, Timeline
from manimx.objects import RNode


def by_name(snapshots: Iterable[RNode], roots: Iterable[Mobject]) -> dict[str, RNode]:
    """``snapshots`` keyed by the names of the Mobjects in ``roots``' families."""
    names = {mobject.uid: mobject.name for root in roots for mobject in root.family()}
    return {names[snapshot.uid]: snapshot for snapshot in snapshots}


def timeline_roots(timeline: Timeline) -> list[Mobject]:
    """The root Mobjects of every clip's scene."""
    return [root for clip in timeline.clips for root in clip.scene.roots]


def clip_names(groups: Iterable[RNode], timeline: Timeline) -> list[str]:
    """The names of the clips whose render groups are ``groups``."""
    names = {clip.uid: clip.name for clip in timeline.clips}
    return [names[group.uid] for group in groups]
