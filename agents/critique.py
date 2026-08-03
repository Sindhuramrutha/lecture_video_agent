"""Critique: an LLM vision check on a rendered clip against its brief.

Judges sampled stills, not motion — good enough to catch broken layout,
off-screen objects, illegible text, or a clip that ignores the brief.
This is the automated half of the render loop; a human reviews the
clip that survives it (or exhausts retries) at the approval gate.
"""

import json
from pathlib import Path

from utils.llm import generate_with_images
from utils.video_frames import extract_frames

SYSTEM_PROMPT = """You are reviewing a rendered Manim animation clip for a STEM educational \
video, checking it against its brief. You will see a few frames sampled evenly from start to \
end of the clip, plus the animation brief and narration it needs to match.

Check for:
- Broken rendering: overlapping text, objects off-screen, empty/blank frames, illegible text.
- Whether the frames plausibly show the described visual progression (judge intent and \
structure from stills, not fine motion).
- Whether it looks like a clean, professional STEM education animation in the style of \
3Blue1Brown.

Respond with ONLY a JSON object, no markdown fences, no commentary, in this exact shape:
{"passed": true, "feedback": "specific, actionable feedback if failed, else empty string"}
"""


def critique_render(video_path: Path, render_prompt: str, narration: str) -> tuple[bool, str]:
    frames = extract_frames(video_path, n=3)
    user_prompt = f"Animation brief:\n{render_prompt}\n\nNarration:\n{narration}\n"
    raw = generate_with_images(SYSTEM_PROMPT, user_prompt, frames, max_tokens=1024)
    data = json.loads(raw)
    return data["passed"], data.get("feedback", "")
