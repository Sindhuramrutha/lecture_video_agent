"""Central place to read environment configuration.

Keeping this as one small module means every other part of the
codebase reads config the same way, and we only touch one file
when a new setting is needed.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class Settings(BaseModel):
    anthropic_api_key: str
    llm_model: str = "claude-sonnet-5"
    tts_voice: str = "en_US-lessac-medium"


@lru_cache
def get_settings() -> Settings:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return Settings(
        anthropic_api_key=api_key,
        llm_model=os.environ.get("LLM_MODEL", "claude-sonnet-5"),
        tts_voice=os.environ.get("TTS_VOICE", "en_US-lessac-medium"),
    )
