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
    llm_base_url: str = "http://localhost:8800/v1"
    llm_api_key: str = "not-needed"
    llm_model: str = "qwen2.5-72b-instruct-awq"
    tts_voice: str = "af_heart"
    # Code-generation model (scene_coder only) — separate endpoint so it can point
    # at a different provider (e.g. DeepSeek) than scene_script/prompt_maker.
    code_llm_base_url: str = "http://localhost:8800/v1"
    code_llm_api_key: str = "not-needed"
    code_llm_model: str = "qwen2.5-72b-instruct-awq"


@lru_cache
def get_settings() -> Settings:
    llm_base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8800/v1")
    llm_api_key = os.environ.get("LLM_API_KEY", "not-needed")
    llm_model = os.environ.get("LLM_MODEL", "qwen2.5-72b-instruct-awq")
    return Settings(
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        tts_voice=os.environ.get("TTS_VOICE", "af_heart"),
        # Falls back to the default LLM settings if CODE_LLM_* isn't set.
        code_llm_base_url=os.environ.get("CODE_LLM_BASE_URL", llm_base_url),
        code_llm_api_key=os.environ.get("CODE_LLM_API_KEY", llm_api_key),
        code_llm_model=os.environ.get("CODE_LLM_MODEL", llm_model),
    )
