"""Single entry point for text-to-speech in the pipeline.

Every stage calls `synthesize()` instead of touching a TTS engine
directly. Swapping Piper for Fish Speech (or a cloud engine) later
means changing this file only. Which Piper voice to use is set via
TTS_VOICE in .env — see assets/voices/ for what's downloaded.
"""

import wave
from pathlib import Path

from piper import PiperVoice

from config.settings import get_settings

VOICES_DIR = Path(__file__).resolve().parent.parent / "assets" / "voices"

_voice: PiperVoice | None = None
_loaded_voice_name: str | None = None


def _get_voice() -> PiperVoice:
    global _voice, _loaded_voice_name
    voice_name = get_settings().tts_voice
    if _voice is None or _loaded_voice_name != voice_name:
        voice_path = VOICES_DIR / f"{voice_name}.onnx"
        _voice = PiperVoice.load(str(voice_path))
        _loaded_voice_name = voice_name
    return _voice


def synthesize(text: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    voice = _get_voice()
    with wave.open(str(out_path), "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    return out_path
