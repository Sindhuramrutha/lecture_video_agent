"""Detailed Scripter agent: turns an approved teaching plan into full textbook-style content.

The LLM decides wording, which equations to derive, which analogies and
examples to use, grounded in any reference material supplied. This is
content authority, not a spoken script — it does not decide narration
phrasing, visuals, or scene staging. The Audio-Visual Scene Script
agent adapts this content into spoken narration and scenes.
"""

from state import VideoState
from utils.llm import generate
from utils.notify import notify
from utils.revision import build_revision_prompt

SYSTEM_PROMPT = """You are an expert textbook author for STEM topics, writing in a rigorous \
but accessible style — think a good undergraduate textbook chapter, not a lecture transcript.

Given a topic, audience, learning objectives, an approved teaching plan, and optional \
reference material, write the detailed content for this chapter, organized by subtopic \
(matching the teaching plan's sections).

For each subtopic, write:
- A section heading, and subsection headings where the content naturally divides.
- The full explanatory content: definitions, derivations, and reasoning, written as textbook \
prose (not spoken narration — this is a reference document, not a script).
- Every equation, in LaTeX, with enough surrounding explanation that a reader could follow the \
derivation.
- A worked example wherever one would clarify the concept — use judgment on which subtopics \
actually need one based on the topic, not every subtopic requires it.
- Every analogy used to build intuition, clearly marked as intuition rather than rigorous \
argument.

If reference material is provided, ground your explanations, terminology, and examples in it \
— do not contradict it.

This is content only — do not describe animations, visuals, timing, or spoken delivery. That \
is decided in a later stage from this content.

Respond with the content as plain markdown text, organized under a heading per subtopic, with \
subsections as needed. Do not wrap the response in code fences. Do not include JSON.
"""


def _base_prompt(state: VideoState) -> str:
    objectives_block = "\n".join(f"- {o}" for o in state.objectives)
    references_block = (
        "\n\n".join(
            f"--- Reference {i + 1} ---\n{ref}" for i, ref in enumerate(state.references)
        )
        if state.references
        else "(none provided)"
    )
    return (
        f"Topic: {state.topic}\n"
        f"Audience: {state.audience}\n"
        f"Target duration: {state.duration_minutes} minutes\n\n"
        f"Learning objectives:\n{objectives_block}\n\n"
        f"Approved teaching plan:\n{state.teaching_plan}\n\n"
        f"Reference material:\n{references_block}\n"
    )


def detailed_scripter_node(state: VideoState) -> dict:
    detailed_script = generate(system=SYSTEM_PROMPT, prompt=_base_prompt(state), max_tokens=8192)
    notify("Detailed Scripter done — review the content")
    return {"detailed_script": detailed_script}


def revise_script(state: VideoState, instruction: str) -> dict:
    prompt = build_revision_prompt(_base_prompt(state), state.detailed_script, instruction)
    detailed_script = generate(system=SYSTEM_PROMPT, prompt=prompt, max_tokens=8192)
    return {"detailed_script": detailed_script}
