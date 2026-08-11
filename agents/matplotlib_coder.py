"""Matplotlib Code Generator: writes a self-contained animation script for
data-driven scenes — real computed numbers, not stylized geometry.

Same retry contract as scene_coder.py: on retry, previous code and the
specific critique feedback are included so the fix targets the actual
problem.
"""

from typing import Optional

from utils.llm import generate_code_model, strip_code_fences

SYSTEM_PROMPT = """You are an expert Matplotlib developer writing data-driven animations for \
STEM educational videos, in the style of 3Blue1Brown: clean, readable, uncluttered.

Given a timed animation brief, write a single self-contained Python script that renders it as \
an MP4 using matplotlib.animation.FuncAnimation. Compute any data the brief needs (e.g. \
numerically integrate an ODE) directly in the script with numpy — don't fabricate canned \
numbers if the brief implies a real calculation.

Visual style — this is why a scene is routed to matplotlib instead of Manim, so make it count. \
Matplotlib scenes play back-to-back with Manim scenes in the same video, so they must look like \
the same course, not a different tool: the audience spans children through adults, so vivid, \
high-contrast color reads better than a flat default palette for everyone.
- Background: exactly `"#0F1116"` — `fig.patch.set_facecolor("#0F1116")` and \
`ax.set_facecolor("#0F1116")` — this is the same dark background every Manim scene in this course \
uses; never the default white-background/blue-line look, and never a different dark shade.
- Light gridlines at low alpha, a clean sans-serif-adjacent font via matplotlib's defaults, axis \
labels and tick text in white or light grey for readability against the dark background.
- For a single line/waveform, prefer coloring it by a colormap along its own length (build a \
`LineCollection` from consecutive point-pair segments, color each segment by position or by \
signal value with `cmap`, e.g. `plt.cm.plasma`/`plt.cm.turbo`) instead of one flat color — this \
reads as far more polished for a waveform, spectrum, or signal plot specifically. For a filled \
region (area under a curve, a highlighted band), use `ax.fill_between` with a vivid, opaque-ish \
color rather than leaving it as an outline only.
- Label axes and any key features (peaks, thresholds, an equilibrium line) directly on the plot \
with `ax.annotate`/`ax.text`, not just a legend box — size the single most important annotation \
noticeably larger than routine axis/tick labels, the same way a key term gets emphasized in a \
Manim scene.
- Keep exactly one plot idea per scene — don't cram multiple unrelated subplots into one scene \
just because the brief mentions several things; if the beats genuinely need multiple axes, use \
`fig.add_subplot`/`gridspec` deliberately, not a cluttered default grid.

Rules:
- Start with `import matplotlib` then `matplotlib.use("Agg")` before any other matplotlib \
import (headless rendering, no display).
- Import only matplotlib, numpy, and the Python standard library.
- Total animation runtime must be within 1 second of the target duration.
- Minimum output resolution is 1280x720 (720p) -- never below this. `figure size in inches × dpi` \
determines the pixel dimensions, so pick both together to guarantee the floor, e.g. \
`fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)` (exactly 1280x720) or any other \
combination whose product is at least that on both axes — a small figsize at a low dpi (e.g. \
`figsize=(6, 4), dpi=100`, only 600x400) silently fails this floor.
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
    previous_scene_code: Optional[str] = None,
    chapter_context: Optional[str] = None,
) -> str:
    user_prompt = f"Animation brief:\n{render_prompt}\n\nTarget duration: {duration:.2f} seconds"
    if chapter_context:
        user_prompt += (
            f"\n\nFor situational awareness, here is the whole chapter this scene belongs to "
            f"(subtopic + narration for every scene) — use it to keep this scene's content, tone, "
            f"and visual choices (axis ranges, labels, colors) consistent with where it sits in "
            f"the larger narrative, not just the isolated brief above:\n{chapter_context}"
        )
    if previous_scene_code:
        user_prompt += (
            f"\n\nThe previous scene's code (for continuity — reuse its exact values/styling "
            f"where the brief refers to the same data/object, not new ones you make up):\n"
            f"{previous_scene_code}"
        )
    if previous_code and feedback:
        user_prompt += (
            f"\n\nYour previous attempt had this problem:\n{feedback}\n\n"
            f"Previous code:\n{previous_code}\n\nFix it."
        )
    return strip_code_fences(generate_code_model(system=SYSTEM_PROMPT, prompt=user_prompt, max_tokens=65536))
