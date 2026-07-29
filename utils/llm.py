"""Single entry point for every LLM call in the system.

Every agent calls `generate()` instead of touching a provider SDK
directly. Swapping Claude for a self-hosted model later means
changing this file only.
"""

from anthropic import Anthropic

from config.settings import get_settings


def generate(system: str, prompt: str, max_tokens: int = 4096) -> str:
    settings = get_settings()
    client = Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
