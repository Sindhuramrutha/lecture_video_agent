"""Planner agent: turns a topic/audience/duration into a teaching plan.

The LLM decides pedagogy (objectives, structure, sequencing). This
module only builds the prompt, calls the LLM, and parses the result
into the state — it makes no pedagogical decisions itself.
"""

import json

from state import VideoState
from utils.llm import generate
from utils.notify import notify
from utils.revision import build_revision_prompt

SYSTEM_PROMPT = """You are an expert curriculum designer for STEM educational videos, \
in the style of 3Blue1Brown: rigorous, clear, and built around a strong narrative arc \
rather than a dry list of facts.

Given a topic, audience, and target duration, produce a teaching plan.

Respond with ONLY a JSON object, no markdown fences, no commentary, in this exact shape:
{
  "objectives": ["objective 1", "objective 2", ...],
  "teaching_plan": "a structured plan as plain text, with sections and pacing notes"
}

Guidelines:
- 3 to 6 learning objectives, each a single concrete sentence.
- The teaching plan should sequence concepts so each builds on the last, \
name the key intuition/visual for each section, and include rough time \
allocations that sum to approximately the target duration.
"""


def _base_prompt(state: VideoState) -> str:
    return (
        f"Topic: {state.topic}\n"
        f"Audience: {state.audience}\n"
        f"Target duration: {state.duration_minutes} minutes\n"
    )


def plan_node(state: VideoState) -> dict:
    raw = generate(system=SYSTEM_PROMPT, prompt=_base_prompt(state))
    data = json.loads(raw)
    notify("Planner done — review the plan")
    return {
        "objectives": data["objectives"],
        "teaching_plan": data["teaching_plan"],
    }


def revise_plan(state: VideoState, instruction: str) -> dict:
    previous_output = json.dumps(
        {"objectives": state.objectives, "teaching_plan": state.teaching_plan}
    )
    prompt = build_revision_prompt(_base_prompt(state), previous_output, instruction)
    raw = generate(system=SYSTEM_PROMPT, prompt=prompt)
    data = json.loads(raw)
    return {
        "objectives": data["objectives"],
        "teaching_plan": data["teaching_plan"],
    }
