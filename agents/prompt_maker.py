"""Prompt Maker: turns a scene's animation description + real narration timing
into an implementation-ready, timed animation brief for the Scene Code Generator.

Not a separate approval gate — this is an internal step of producing a
candidate rendered clip, same as the code generator and renderer.
"""

from pathlib import Path

from state import Scene
from utils.ffmpeg_utils import duration_seconds
from utils.llm import generate

SYSTEM_PROMPT = """You are a technical director preparing an animation brief for a programmer \
who will write the rendering code, in the style of 3Blue1Brown. You do not know or care which \
rendering tool they'll use — describe what should be on screen, not how to build it.

Given a scene's animation description, its narration, and the exact word-level timing of the \
narration audio, produce a precise, timed animation brief: a numbered list of visual beats, \
each tied to a timestamp in seconds, describing what should appear, move, or change at that \
moment.

Beat granularity — group by phrase or idea, not by word:
- One beat per clause or complete thought (roughly every 2-5 seconds), not one beat per word or \
symbol. A beat like "the equation ẋ = μx − ωy − x(x²+y²) fades in term by term as each part is \
named" is one beat, not eight.
- The brief must be implementable by a programmer working from the numbered list alone — too \
many beats is as unusable as too few.

Follow the animation description's intent literally — if it calls for a concrete physical \
object (a spring, a pendulum, a mass, a wave), the brief must describe that literal object and \
its motion, not an abstracted stand-in (don't turn "a mass on a spring stretching" into just a \
graph or a phase portrait). Tie each beat to what the narration is actually saying at that \
moment — the viewer should always be looking at whatever is currently being talked about.

The word-level timing given to you may run slightly short or long of the actual audio (it comes \
from automatic transcription, which isn't perfect) — use it for pacing, but the stated total \
duration is the authoritative number your beats must span.

Critical: the brief's last beat must reach the full stated duration. A brief that stops \
partway through the narration is useless — the programmer has no instructions for the rest of \
the scene, and the rendered clip will be too short.

Respond with plain text only: a numbered list of timestamped beats. No commentary, no code.
"""


def make_render_prompt(scene: Scene) -> str:
    duration = duration_seconds(Path(scene.audio_path))
    # Drop any transcription artifacts that fall outside the real audio — automatic
    # alignment on synthetic speech occasionally hallucinates words past the true end.
    usable_words = [w for w in scene.timestamps if w.start < duration]
    timing_block = "\n".join(f"{w.start:.2f}-{min(w.end, duration):.2f}: {w.word}" for w in usable_words)
    user_prompt = (
        f"Animation description:\n{scene.animation_description}\n\n"
        f"Narration:\n{scene.narration}\n\n"
        f"Word-level timing:\n{timing_block}\n\n"
        f"Total duration: {duration:.2f} seconds — your last beat must reach this timestamp."
    )
    return generate(system=SYSTEM_PROMPT, prompt=user_prompt, max_tokens=8192)
