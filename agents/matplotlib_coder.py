"""Matplotlib Code Generator: writes a self-contained animation script for
data-driven scenes — real computed numbers, not stylized geometry.

Same retry contract as scene_coder.py: on retry, previous code and the
specific critique feedback are included so the fix targets the actual
problem.
"""

from typing import Optional

from utils.llm import generate

SYSTEM_PROMPT = """You are an expert Matplotlib developer writing data-driven animations for \
STEM educational videos, in the style of 3Blue1Brown: clean, readable, uncluttered.

Given a timed animation brief, write a single self-contained Python script that renders it as \
an MP4 using matplotlib.animation.FuncAnimation. Compute any data the brief needs (e.g. \
numerically integrate an ODE) directly in the script with numpy — don't fabricate canned \
numbers if the brief implies a real calculation.

Rules:
- Start with `import matplotlib` then `matplotlib.use("Agg")` before any other matplotlib \
import (headless rendering, no display).
- Import only matplotlib, numpy, and the Python standard library.
- Total animation runtime must be within 1 second of the target duration.
- Save the result with exactly this call: `anim.save("output.mp4", writer="ffmpeg", fps=30)` — \
that exact filename, nothing else.
- No file I/O other than that save call, no network access, no external assets.
- Respond with ONLY the Python code. No markdown fences, no commentary.
"""


def generate_code(
    render_prompt: str,
    duration: float,
    previous_code: Optional[str] = None,
    feedback: Optional[str] = None,
) -> str:
    user_prompt = f"Animation brief:\n{render_prompt}\n\nTarget duration: {duration:.2f} seconds"
    if previous_code and feedback:
        user_prompt += (
            f"\n\nYour previous attempt had this problem:\n{feedback}\n\n"
            f"Previous code:\n{previous_code}\n\nFix it."
        )
    return generate(system=SYSTEM_PROMPT, prompt=user_prompt, max_tokens=16384)
