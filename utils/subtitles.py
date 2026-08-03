"""Builds an .srt file directly from the word-alignment timestamps already
produced by the Audio Generator stage — not a second ASR pass on the
final mix. Same source of truth that drives animation timing, so
subtitles, audio, and video all stay in sync by construction.
"""

from pathlib import Path

from state import Scene, WordTiming
from utils.ffmpeg_utils import duration_seconds

MAX_WORDS_PER_CUE = 8
MAX_CUE_SECONDS = 4.0


def _group_words(words: list[WordTiming]) -> list[list[WordTiming]]:
    groups = []
    current: list[WordTiming] = []
    for word in words:
        current.append(word)
        span = current[-1].end - current[0].start
        if len(current) >= MAX_WORDS_PER_CUE or span >= MAX_CUE_SECONDS:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _format_timestamp(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int(round((secs - int(secs)) * 1000))
    return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d},{millis:03d}"


def build_srt(scenes: list[Scene]) -> str:
    cues = []
    offset = 0.0
    index = 1
    for scene in scenes:
        duration = duration_seconds(Path(scene.audio_path))
        # Drop transcription artifacts beyond the real clip — otherwise a stray cue can
        # land inside the *next* scene's composed video once offsets are added up.
        usable_words = [w for w in scene.timestamps if w.start < duration]
        for group in _group_words(usable_words):
            start = offset + group[0].start
            end = offset + min(group[-1].end, duration)
            text = " ".join(w.word for w in group)
            cues.append(
                f"{index}\n{_format_timestamp(start)} --> {_format_timestamp(end)}\n{text}\n"
            )
            index += 1
        offset += duration
    return "\n".join(cues)
