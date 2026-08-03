"""Audio-Visual Scene Script agent: turns textbook content into spoken scenes.

The LLM decides scene boundaries, writes the actual spoken narration
(adapting the textbook-style detailed script into something meant to
be heard, not read), the animation for each scene, and which renderer
it needs. It makes no rendering decisions itself — that happens in the
Visual branch, which dispatches each scene to its chosen renderer
(Manim or Matplotlib, for now).
"""

import json

from state import Scene, VideoState
from utils.llm import generate
from utils.notify import notify
from utils.revision import build_revision_prompt

SYSTEM_PROMPT = """You are an expert video director for STEM educational videos, in the style \
of 3Blue1Brown.

You are given the detailed content for this video, written as textbook prose (sections, \
subsections, equations, worked examples). Break it into a sequence of short scenes. For each \
scene, decide:
- The spoken narration for that scene: adapt the textbook content into natural spoken English \
— conversational, paced for listening, not read verbatim from the page. Cover the same ideas, \
equations, and examples the content contains for that part, in the same order, without \
inventing new facts, claims, or examples beyond what the content supports.
- A concrete description of the animation: what is drawn, what moves, what appears or \
disappears, in enough detail that a programmer could implement it without seeing the rest of \
the script.
- The scene type, based on this rule — pick the cheapest option that still teaches the point, \
don't build an animation just because you can:
  - "manim": stylized/constructed visuals — equations, phase portraits, vector fields, \
geometric constructions, code walkthroughs, or a literal physical system (a spring, a \
pendulum, a mass, a wave, a circuit) built from scratch.
  - "matplotlib": the scene's point is real computed data — an actual numeric time series, a \
histogram, a statistical plot, the literal output of integrating an equation — where showing \
genuine numbers/data matters more than stylized geometry.
  - "image_only": a single static image or icon on screen (a photo, a diagram, an illustrative \
SVG) teaches the point with no real animation needed.
  - "audio_only": no meaningful visual is needed beyond a plain/branded background — a \
transition, a rhetorical aside, a moment of pure exposition.
  Prefer "image_only" or "audio_only" over "manim"/"matplotlib" whenever a full animation \
wouldn't add teaching value. Default to "manim" when a scene could reasonably be either.

Critical rule for "manim"/"matplotlib" scenes — depict concretely, not just abstractly:
- If the content is building intuition for a concept that has a physical embodiment (a \
harmonic oscillator as a mass on a spring or a swinging pendulum, a wave as a moving string, \
etc.), the animation must show that literal physical object in motion, not just its abstract \
math representation (equations, phase portraits). Show the concrete picture first when a \
concept is first introduced; abstract diagrams (phase planes, graphs) are for scenes that are \
explicitly analyzing the math, once the concrete picture is established.
- The animation must track what the narration is actually saying at each moment — describe \
what's on screen scene beat by scene beat, matching the narration's own sequence, not a \
generic summary of the topic.

Scenes should be short, roughly 15-45 seconds of spoken narration each. Cover the entire \
content with no gaps and no overlaps, in order.

Respond with ONLY a JSON array, no markdown fences, no commentary, in this exact shape:
[
  {"subtopic": "...", "narration": "...", "animation_description": "...", "scene_type": "manim"},
  ...
]
"""


def _base_prompt(state: VideoState) -> str:
    return (
        f"Topic: {state.topic}\n"
        f"Audience: {state.audience}\n\n"
        f"Detailed content:\n{state.detailed_script}\n"
    )


def _parse_scenes(raw: str) -> list[Scene]:
    data = json.loads(raw)
    return [
        Scene(
            id=i,
            subtopic=item["subtopic"],
            narration=item["narration"],
            animation_description=item["animation_description"],
            scene_type=item["scene_type"],
        )
        for i, item in enumerate(data)
    ]


MAX_JSON_ATTEMPTS = 3


def _generate_scenes(system: str, prompt: str) -> list[Scene]:
    # Long, technical outputs occasionally slip on strict JSON syntax (a stray "="
    # instead of ": ", a dropped comma). Rare and stochastic — retrying the same
    # call is far more reliable than trying to string-patch malformed JSON.
    last_error = None
    for _ in range(MAX_JSON_ATTEMPTS):
        raw = generate(system=system, prompt=prompt, max_tokens=16384)
        try:
            return _parse_scenes(raw)
        except (json.JSONDecodeError, KeyError) as exc:
            last_error = exc
    raise last_error


def scene_script_node(state: VideoState) -> dict:
    scenes = _generate_scenes(SYSTEM_PROMPT, _base_prompt(state))
    notify("Scene Script done — review the scenes")
    return {"scenes": scenes}


def revise_scenes(state: VideoState, instruction: str) -> dict:
    previous_output = json.dumps([s.model_dump() for s in state.scenes])
    prompt = build_revision_prompt(_base_prompt(state), previous_output, instruction)
    scenes = _generate_scenes(SYSTEM_PROMPT, prompt)
    return {"scenes": scenes}
