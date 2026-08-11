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
subsections, equations, worked examples). Break it into a sequence of short scenes.

Fidelity to the source — the single most important rule, above pacing, engagement, or scene \
count:
- Every fact, number, equation, example, and claim in the narration must be traceable to the \
detailed content given to you. Do not invent additional facts, statistics, historical \
anecdotes, "fun" trivia, or analogies that are not in the source, no matter how plausible or \
pedagogically appealing they sound.
- Do not add unscientific claims, folk explanations, or narrative flourishes to make a scene \
sound more engaging. Accuracy and grounding in the source always outrank color or storytelling.
- If the source material for a subtopic is thin, write a short scene rather than padding it \
with invented content to fill time or hit the target duration.
- Not inventing content is different from compressing it — by default, do not summarize the \
source down. If the source explains *why* something is true, works through an example, gives \
an aside, or adds a nuance, narrate all of that in full, in natural spoken words. Adapting to \
spoken style means changing phrasing and pacing, not shortening or skipping explanation that's \
already there.
- The content may state its own target duration or timeline (e.g. a recommended length) — if \
it does, that is the authoritative target, in addition to any target given separately. Compare \
it against a full, uncompressed narration of the content:
  - If the full narration would come in *under* the target, that is fine — do not pad, invent, \
or slow down to reach it. A shorter-than-target video from thin source material is correct.
  - If the full narration would run *over* the target, then and only then trim it down to fit \
— cut redundant restatements, secondary asides, or less-essential elaboration first, keeping \
every core fact, equation, and example the content builds around. Never invent or embellish to \
fill time; only ever trim to hit an over-budget timeline, never pad to hit an under-budget one.

For each scene, decide:
- The spoken narration for that scene: rephrase the textbook content into natural spoken \
English — conversational, paced for listening, rather than read verbatim off the page — while \
preserving its full depth. Cover every idea, equation, example, and explanatory detail the \
content contains for that part, in the same order, at the same level of depth as the source.
- A concrete description of the animation: what is drawn, what moves, what appears or \
disappears, in enough detail that a programmer could implement it without seeing the rest of \
the script.
- The scene type, based on this rule — pick the cheapest option that still teaches the point, \
don't build an animation just because you can:
  - "manim": stylized or constructed scenes — equations, phase portraits, vector fields, \
geometric constructions, code walkthroughs, a literal physical system (a mass on a spring, a \
swinging pendulum) in motion. Anything built from shapes and motion rather than plotted from \
real numbers.
  - "matplotlib": real computed data rendered as an actual plot — a numeric time series, a \
waveform, a histogram, a statistical or scientific chart, the literal output of integrating an \
equation. Prefer this over "manim" whenever the scene's whole point is showing real data \
precisely plotted (axes, gridlines, a properly scaled curve) rather than a stylized diagram — \
matplotlib renders this kind of thing more precisely and cleanly than hand-built Manim shapes.
  - "image_only": a single static image or icon on screen teaches the point with no real \
animation needed. Important: no real illustration generator exists yet for this scene type — it \
currently renders as a plain wrapped-text title card, not an actual picture. Only ever choose \
"image_only" for a genuinely brief moment (a few seconds, well under this course's normal \
15-45s scene length) where that's an acceptable stand-in — never for a scene with substantial \
narration. A static card sitting on screen for 20-30+ seconds while narration keeps talking is a \
worse viewing experience than an animated scene would have been; if the narration for this scene \
runs anywhere near the normal scene length, use "manim" or "matplotlib" instead, even if the \
visual is simple.
  - "audio_only": no meaningful visual is needed beyond a plain/branded background — a \
transition, a rhetorical aside, a moment of pure exposition. Same duration caveat as \
"image_only" — reserve for genuinely brief moments, not scenes with substantial narration.
  Prefer "image_only" or "audio_only" over an animated scene whenever a full animation wouldn't \
add teaching value. Default to "manim" or "matplotlib" (whichever fits per the rules above) \
when a scene needs any animation at all.

Critical rule for "manim" scenes — depict concretely, not just abstractly:
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
        f"Audience: {state.audience}\n"
        f"Target total duration: approximately {state.duration_minutes} minutes across all "
        f"scenes combined — use this to guide how many scenes you produce and how much each "
        f"covers, not as a per-scene figure.\n\n"
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
        raw = generate(system=system, prompt=prompt, max_tokens=8192)
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
