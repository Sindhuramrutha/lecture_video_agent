"""Planner agent: turns a topic/audience/duration into a teaching plan.

The LLM decides pedagogy (objectives, structure, sequencing). This
module only builds the prompt, calls the LLM, and parses the result
into the state — it makes no pedagogical decisions itself.
"""

import json

from state import VideoState
from utils.llm import generate

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


def plan_node(state: VideoState) -> dict:
    user_prompt = (
        f"Topic: {state.topic}\n"
        f"Audience: {state.audience}\n"
        f"Target duration: {state.duration_minutes} minutes\n"
    )
    raw = generate(system=SYSTEM_PROMPT, prompt=user_prompt)
    data = json.loads(raw)
    return {
        "objectives": data["objectives"],
        "teaching_plan": data["teaching_plan"],
    }
