"""Shared helper for building a 'revise given human feedback' prompt.

Used by every stage's revise_* function so a human's free-text
instruction ("use a pendulum instead", "make this punchier") turns
into one regeneration call — same idea as the code-retry feedback loop
in scene_coder.py, just with a human as the source of feedback instead
of a render error or auto-critique.
"""


def build_revision_prompt(base_prompt: str, previous_output: str, instruction: str) -> str:
    return (
        f"{base_prompt}\n\n"
        f"--- Previous output ---\n{previous_output}\n\n"
        f"--- Requested change ---\n{instruction}\n\n"
        "Revise the output to address the requested change. Keep everything else the same "
        "unless the change implies otherwise. Respond in the exact same format specified above."
    )
