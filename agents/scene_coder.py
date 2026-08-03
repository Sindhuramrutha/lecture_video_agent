"""Scene Code Generator: writes Manim code implementing a scene's render brief.

On retry (after a failed critique), the previous code and the specific
feedback are included so the LLM fixes the actual problem rather than
starting over blind.
"""

from typing import Optional

from utils.llm import generate

SYSTEM_PROMPT = """You are an expert Manim (Community Edition, v0.20+) developer for STEM \
educational animations, in the style of 3Blue1Brown: clean, readable, uncluttered.

Given a timed animation brief, write a single self-contained Manim Scene subclass that \
implements it as closely as Manim allows.

Manim is fully capable of literal physical simulations, not just graphs and equations — build \
what the brief actually describes. A spring is a coil (e.g. a zigzag/helical Line or a custom \
VMobject) that stretches and compresses; a pendulum is a Line (rod) plus a Dot or Circle (bob) \
rotating about a pivot, animated with UpdateFromFunc or a ValueTracker; a mass is a Square or \
Circle. If the brief calls for one of these, build the literal object — do not substitute an \
abstract diagram instead.

Rules:
- Import only from `manim` (`from manim import *`) plus the Python standard library and numpy.
- The class must be named exactly `{class_name}`.
- Total animation runtime must be within 1 second of the target duration. Use self.wait() to \
fill gaps.
- No file I/O, no network access, no external assets.
- Respond with ONLY the Python code. No markdown fences, no commentary.
"""


def generate_code(
    class_name: str,
    render_prompt: str,
    duration: float,
    previous_code: Optional[str] = None,
    feedback: Optional[str] = None,
) -> str:
    system = SYSTEM_PROMPT.format(class_name=class_name)
    user_prompt = f"Animation brief:\n{render_prompt}\n\nTarget duration: {duration:.2f} seconds"
    if previous_code and feedback:
        user_prompt += (
            f"\n\nYour previous attempt had this problem:\n{feedback}\n\n"
            f"Previous code:\n{previous_code}\n\nFix it."
        )
    return generate(system=system, prompt=user_prompt, max_tokens=16384)
