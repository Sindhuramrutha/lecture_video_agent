"""Word-level timestamps for generated narration audio.

Uses Whisper ASR on the synthesized audio rather than true forced
alignment against the known script text — simpler, no extra
dependency, and good enough for scene timing. Domain-specific terms
can be mis-transcribed (timing stays roughly right even when the word
label is wrong); switch to a forced aligner (e.g. whisperx) if that
becomes a real problem for subtitles later.
"""

from pathlib import Path

from faster_whisper import WhisperModel

from state import WordTiming

_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel("base.en", device="cpu", compute_type="int8")
    return _model


def align(audio_path: Path) -> list[WordTiming]:
    model = _get_model()
    segments, _ = model.transcribe(str(audio_path), word_timestamps=True)
    return [
        WordTiming(word=word.word.strip(), start=word.start, end=word.end)
        for segment in segments
        for word in segment.words
    ]
