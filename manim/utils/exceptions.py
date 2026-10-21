from __future__ import annotations

__all__ = [
    "EndSceneEarlyException",
    "MultiAnimationOverrideException",
]


class EndSceneEarlyException(Exception):
    pass


class MultiAnimationOverrideException(Exception):
    pass
