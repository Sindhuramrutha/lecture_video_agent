"""Expands a human's rough edit back to full pipeline-quality detail.

Used when reloading a manually-edited stage file: a terse edit like
"add a sine wave graph" gets expanded to the same level of detail the
agent would have produced, so downstream stages have enough to work
with. Already-detailed edits pass through unchanged.
"""

from utils.llm import generate

SYSTEM_PROMPT = """You expand a human's rough edit into the same level of detail and style \
as the surrounding pipeline content, without changing their intent.

Rules:
- If a piece of text already reads like a complete, detailed version, leave it unchanged.
- If it reads like a rough note or shorthand, expand it into full detail matching the style \
described in the context given.
- Never add ideas, claims, or scope the human didn't imply. Expand clarity and detail, not \
content.
- If the input is JSON, return valid JSON with the exact same structure and keys. Only expand \
string values that read like rough notes — leave short categorical fields (like an id or a \
type enum) untouched.
- Respond with ONLY the resulting text (or JSON). No commentary, no markdown fences, no preamble.
"""


def enhance(raw_text: str, context: str) -> str:
    user_prompt = f"Context: {context}\n\nText to check/expand:\n{raw_text}"
    return generate(system=SYSTEM_PROMPT, prompt=user_prompt, max_tokens=16384)
