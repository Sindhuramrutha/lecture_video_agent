"""Single entry point for every LLM call in the system.

Every agent calls `generate()` or `generate_with_images()` instead of
touching a provider SDK directly. Swapping Claude for a self-hosted
model later means changing this file only.
"""

import base64
from pathlib import Path

from anthropic import Anthropic

from config.settings import get_settings


def _client() -> Anthropic:
    settings = get_settings()
    return Anthropic(api_key=settings.anthropic_api_key)


def _text_from(content) -> str:
    return "".join(block.text for block in content if block.type == "text")


def generate(system: str, prompt: str, max_tokens: int = 4096) -> str:
    settings = get_settings()
    response = _client().messages.create(
        model=settings.llm_model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return _text_from(response.content)


def generate_with_images(
    system: str, prompt: str, image_paths: list[Path], max_tokens: int = 1024
) -> str:
    settings = get_settings()
    content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.standard_b64encode(Path(p).read_bytes()).decode("utf-8"),
            },
        }
        for p in image_paths
    ]
    content.append({"type": "text", "text": prompt})

    response = _client().messages.create(
        model=settings.llm_model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": content}],
    )
    return _text_from(response.content)
