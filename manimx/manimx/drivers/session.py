"""The timeline's sound for editing in an audio or video program.

    export_session(timeline, video, start=0.0, end=None, sample_rate=48000)
    # next to the video renders/intro.mp4:
    #   renders/intro-stems/Video 3.wav, Script 1.wav, Audio 1.wav, ...
    #   renders/intro.otio   an OpenTimelineIO timeline of the video and
    #                        every stem (Kdenlive, DaVinci Resolve, ...)

``manimx render --stems`` (and the render dialog's "sound for editing")
renders the video without sound and calls this: the sound is mixed in the
other program, where it can be cleaned, compressed and mastered.

Design decisions:

* **One stem per track.** A stem is a WAV of one track's sounds over the
  whole render (the clips' own volumes and fades applied), every stem as
  long as the video and starting with it, so placing each at zero lines
  everything up in any program. A clip per sound file would keep cuts
  editable but needs every program's idea of cuts, offsets and fades;
  stems need none, and are what a mixing engineer expects.
* **The track's volume is in its stem; mute and solo are not.**
  OpenTimelineIO has no faders, so a stem sounds at its track's volume
  and the mix starts as the preview played it. A muted track (or one not
  heard because others solo) is still written, named "(muted)": the
  program can mute it, and nothing is lost.
* **OpenTimelineIO only.** It is open and Kdenlive, DaVinci Resolve and
  others import it. A Reaper project (.rpp) was dropped: Reaper is not
  free.
* **24-bit WAV.** Every audio program reads it; float WAV is not read
  everywhere, and 16 bits would lose the quiet ends of fades.
* **Written by hand.** OpenTimelineIO is JSON, so no library is needed
  for the little of it used here:
  https://opentimelineio.readthedocs.io/en/latest/tutorials/file-format-specification.html
* **Markers at the clips' starts,** named after them, so the program shows
  where each scene begins.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..animation.audio import DEFAULT_SAMPLE_RATE, write_wav
from ..animation.timing import Timeline


@dataclass(frozen=True)
class Stem:
    """One track's sound, written to ``file``."""

    name: str
    file: Path
    muted: bool
    """Not heard in the preview (muted, or others solo)."""


@dataclass(frozen=True)
class Session:
    """What :func:`export_session` wrote."""

    stems: tuple[Stem, ...]
    otio: Path


def export_session(
    timeline: Timeline,
    video: Path,
    start: float = 0.0,
    end: float | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    fps: float = 60.0,
) -> Session:
    """Write the stems of ``timeline`` from ``start`` to ``end`` and the
    timeline using them and ``video`` (see the module docs)."""
    video = Path(video)
    end = timeline.duration if end is None else min(end, timeline.duration)
    length = max(0.0, end - start)
    folder = video.with_name(f"{video.stem}-stems")
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.wav"):
        old.unlink()  # a stem of a track no longer there must not linger
    soloing = any(track.solo for _, track in timeline.tracks())
    stems = []
    for (kind, index), samples in timeline.stems(start, end, sample_rate).items():
        track = timeline.track(kind, index)
        muted = track.muted or (soloing and not track.solo)
        name = timeline.track_name(kind, index) + (" (muted)" if muted else "")
        file = folder / f"{_file_name(name)}.wav"
        for number in range(2, len(stems) + 2):
            if all(stem.file != file for stem in stems):
                break
            file = folder / f"{_file_name(name)} {number}.wav"  # two tracks of one name
        write_wav(file, samples, sample_rate, bits=24)
        stems.append(Stem(name, file, muted))
    markers = [(clip.start - start, clip.name) for clip in sorted(timeline.clips, key=lambda clip: clip.start) if start <= clip.start < end]
    otio = video.with_suffix(".otio")
    otio.write_text(json.dumps(_otio(video, stems, length, fps, markers), indent=2), encoding="utf-8")
    return Session(tuple(stems), otio)


def _file_name(name: str) -> str:
    """A track name made safe as a file name."""
    return re.sub(r"[^\w .()-]+", "_", name).strip() or "track"


def _otio(video: Path, stems: list[Stem], length: float, fps: float, markers: list[tuple[float, str]]) -> dict[str, Any]:
    """An OpenTimelineIO timeline: a video track and one audio track per stem."""

    def time(seconds: float) -> dict[str, Any]:
        return {"OTIO_SCHEMA": "RationalTime.1", "rate": fps, "value": float(math.floor(seconds * fps + 0.5))}

    def span(at: float, duration: float) -> dict[str, Any]:
        return {"OTIO_SCHEMA": "TimeRange.1", "start_time": time(at), "duration": time(duration)}

    def track(name: str, kind: str, file: Path) -> dict[str, Any]:
        clip = {
            "OTIO_SCHEMA": "Clip.1",
            "name": file.name,
            "source_range": span(0.0, length),
            "media_reference": {
                "OTIO_SCHEMA": "ExternalReference.1",
                "target_url": file.resolve().as_uri(),
                "available_range": span(0.0, length),
                "metadata": {},
            },
            "effects": [],
            "markers": [],
            "metadata": {},
        }
        return {"OTIO_SCHEMA": "Track.1", "name": name, "kind": kind, "source_range": None, "children": [clip], "effects": [], "markers": [], "metadata": {}}

    marks = [
        {"OTIO_SCHEMA": "Marker.2", "name": name, "marked_range": span(at, 0.0), "color": "GREEN", "comment": "", "metadata": {}}
        for at, name in markers
    ]
    stack = {
        "OTIO_SCHEMA": "Stack.1",
        "name": "tracks",
        "source_range": None,
        "children": [track("Video", "Video", video)] + [track(stem.name, "Audio", stem.file) for stem in stems],
        "effects": [],
        "markers": marks,
        "metadata": {},
    }
    return {"OTIO_SCHEMA": "Timeline.1", "name": video.stem, "global_start_time": None, "tracks": stack, "metadata": {}}


__all__ = ["Session", "Stem", "export_session"]
