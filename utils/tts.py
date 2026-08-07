"""Single entry point for text-to-speech in the pipeline.

Every stage calls `synthesize()` instead of touching a TTS engine
directly. Swapping Kokoro for something else later means changing this
file only. Which Kokoro voice to use is set via TTS_VOICE in .env
(e.g. "af_heart", "af_bella" — American English voices; see
https://hf.co/hexgrad/Kokoro-82M for the full voice list).
"""

from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline

from config.settings import get_settings

SAMPLE_RATE = 24000  # Kokoro's native output rate

_pipeline: KPipeline | None = None


def _get_pipeline() -> KPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = KPipeline(lang_code="a")  # "a" = American English
    return _pipeline


def synthesize(text: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline = _get_pipeline()
    voice = get_settings().tts_voice

    chunks = [audio for _, _, audio in pipeline(text, voice=voice)]
    audio = np.concatenate([c.numpy() if hasattr(c, "numpy") else c for c in chunks])
    sf.write(str(out_path), audio, SAMPLE_RATE)
    return out_path
