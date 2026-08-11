"""Single entry point for every LLM call in the system.

Every agent calls `generate()`, `generate_code_model()`, or
`generate_with_images()` instead of touching a provider SDK directly.
Swapping providers means changing env vars, not this file — `generate()`
reads LLM_BASE_URL/LLM_API_KEY/LLM_MODEL (defaults to the local vLLM
server running Qwen2.5-72B-Instruct-AWQ; used by scene_script and
prompt_maker), `generate_code_model()` reads the separate CODE_LLM_*
vars (defaults to DeepSeek; used only by scene_coder, where code
correctness matters most) — see config/settings.py. Anyone without
access to the local Qwen server (it's privately hosted, not reachable
over the network) can point LLM_BASE_URL/LLM_API_KEY/LLM_MODEL at any
OpenAI-compatible hosted API instead in their own .env, no code changes
needed — both entry points already disable "thinking" mode (harmless
no-op against a server that doesn't have the concept, like local vLLM;
required against one that does, like DeepSeek) so either backend works
correctly either way. `generate_with_images()` has no backend and is
disabled — see agents/critique.py, which already isn't wired into the
active pipeline (agents/visual_render.py explains why: the
vision-critique loop was dropped in favor of human review).
"""

from pathlib import Path

from openai import OpenAI

from config.settings import get_settings


def _client() -> OpenAI:
    settings = get_settings()
    return OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)


def _code_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(base_url=settings.code_llm_base_url, api_key=settings.code_llm_api_key)


def generate(system: str, prompt: str, max_tokens: int = 4096) -> str:
    settings = get_settings()
    response = _client().chat.completions.create(
        model=settings.llm_model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        # No-op against local vLLM/Qwen (confirmed: it just ignores this field), but
        # required if LLM_BASE_URL ever points at a reasoning model (e.g. DeepSeek) —
        # otherwise hidden reasoning tokens can silently exhaust max_tokens and return
        # empty content, exactly like generate_code_model() had to guard against.
        extra_body={"thinking": {"type": "disabled"}},
    )
    return response.choices[0].message.content


def generate_code_model(system: str, prompt: str, max_tokens: int = 8192) -> str:
    """Calls the code-generation endpoint (currently DeepSeek), thinking disabled.

    Thinking was briefly re-enabled as an untested hypothesis that DeepSeek's
    reasoning pass would help with multi-step bookkeeping (e.g. tracking a
    cumulative scale factor across several sequential .animate.scale() calls)
    -- but that was never actually verified with a controlled before/after
    test, and it has a real, confirmed latency cost (the reasoning pass runs
    before any visible content, on every single call). Disabled again to get
    that latency back now that other changes (the expanded manim_reference.py,
    richer prompt guidance) may cover the same ground without it. Re-enable
    and test head-to-head if a similar cumulative-tracking bug resurfaces.
    """
    settings = get_settings()
    response = _code_client().chat.completions.create(
        model=settings.code_llm_model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        extra_body={"thinking": {"type": "disabled"}},
    )
    return response.choices[0].message.content


def strip_code_fences(text: str) -> str:
    """Strip a leading/trailing ```lang fence some models add despite being told not to.

    Retry-with-feedback alone isn't reliable here — models have repeated this
    exact mistake across retries rather than self-correcting from the
    resulting SyntaxError, so callers that need raw executable source
    (scene_coder) should run their output through this before using it.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


def generate_with_images(
    system: str, prompt: str, image_paths: list[Path], max_tokens: int = 1024
) -> str:
    raise NotImplementedError(
        "generate_with_images() has no local backend: the local LLM "
        "(Qwen2.5-72B-Instruct-AWQ) is text-only. Vision critique is "
        "currently disabled — see agents/critique.py."
    )
