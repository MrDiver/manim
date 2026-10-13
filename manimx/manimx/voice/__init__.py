"""Voice: transcripts of recordings, their words as events and subtitles.

transcript.py holds the data (text, timed words, event names,
subtitles); alignment.py times the words with Whisper.
"""

from .alignment import align, recognize, transcribe, trim_silence
from .transcript import Subtitle, Transcript, Word, srt, transcript_path

__all__ = ["Subtitle", "Transcript", "Word", "align", "recognize", "srt", "transcribe", "transcript_path", "trim_silence"]
