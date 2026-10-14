"""Voice: scripts, their recorded takes, their words as events and subtitles.

script.py holds the scripts (text in parts, word times, takes);
transcript.py the timed words of a text and its subtitles; alignment.py
times the words of a recording with Whisper.
"""

from .alignment import align, recognize, time_take, transcribe, trim_silence
from .script import Part, Script, Take
from .transcript import Subtitle, Transcript, Word, srt

__all__ = ["Part", "Script", "Subtitle", "Take", "Transcript", "Word", "align", "recognize", "srt", "time_take", "transcribe", "trim_silence"]
