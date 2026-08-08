"""Scene Code Generator: writes Manim code implementing a scene's render brief.

On retry (after a failed critique), the previous code and the specific
feedback are included so the LLM fixes the actual problem rather than
starting over blind.
"""

import re
from typing import Optional

from utils.llm import generate_code_model, strip_code_fences
from utils.manim_reference import MANIM_API_REFERENCE

_FENCE_LINE = re.compile(r"^```\w*[ \t]*$", re.MULTILINE)


def _find_matching_paren(text: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


# Real Manim frame is 14.222 x 8.0 units (config.frame_width/frame_height); a small
# inward epsilon avoids flagging things sitting exactly at the edge due to float error.
_FRAME_GUARD_SOURCE = '''
def _check_frame_bounds(_scene, _mx=7.05, _my=3.95):
    for _mob in _scene.mobjects:
        try:
            if getattr(_mob, "z_index", 0) < 0:
                continue  # background layer -- deliberately full-bleed, not clipped content
            _l, _r = _mob.get_left()[0], _mob.get_right()[0]
            _t, _b = _mob.get_top()[1], _mob.get_bottom()[1]
        except Exception:
            continue  # no points anywhere in this mobject's family -- nothing to check
        _entirely_outside = _l > _mx or _r < -_mx or _b > _my or _t < -_my
        _exceeds = _l < -_mx or _r > _mx or _t > _my or _b < -_my
        if _exceeds and not _entirely_outside:
            _name = getattr(_mob, "text", None) or type(_mob).__name__
            raise RuntimeError(
                f"Frame bounds exceeded: {_name!r} spans x=[{_l:.2f},{_r:.2f}] "
                f"y=[{_b:.2f},{_t:.2f}], outside the safe area "
                f"x=[-{_mx},{_mx}] y=[-{_my},{_my}] -- part of it is being cut off "
                f"at the edge of the frame."
            )
'''


def _inject_frame_guard(code: str) -> str:
    """Inserts a call to a bounds-checking helper after every top-level
    self.play(...)/self.wait(...) statement, and defines that helper at module
    level. Turns an invisible "content clipped at the frame edge" visual bug into
    an actual render failure with a specific, actionable message -- the prompt's
    own "stay within ±6.5/±3.5" instruction is a suggestion the model can still
    get wrong (confirmed: it did); this is an enforced, unconditional check that
    feeds the same automatic retry loop as every other render failure.
    """
    out = []
    i = 0
    markers = ("self.play(", "self.wait(")
    while i < len(code):
        # find the earliest next occurrence of either marker
        next_idx, next_marker = -1, None
        for marker in markers:
            idx = code.find(marker, i)
            if idx != -1 and (next_idx == -1 or idx < next_idx):
                next_idx, next_marker = idx, marker
        if next_idx == -1:
            out.append(code[i:])
            break
        open_paren = next_idx + len(next_marker) - 1
        close_paren = _find_matching_paren(code, open_paren)
        if close_paren == -1:
            out.append(code[i:])
            break
        line_start = code.rfind("\n", 0, next_idx) + 1
        indent = code[line_start:next_idx]
        line_end = code.find("\n", close_paren)
        if line_end == -1:
            # last statement in the file, no trailing newline -- without adding one
            # here, the injected call glues onto the same line as the statement
            # (e.g. "self.wait(...)        _check_frame_bounds(self)"), which is a
            # SyntaxError, not a call. Confirmed: this broke a real scene.
            segment = code[i:]
            if not segment.endswith("\n"):
                segment += "\n"
            out.append(segment)
            out.append(f"{indent}_check_frame_bounds(self)\n")
            i = len(code)
            break
        line_end += 1
        out.append(code[i:line_end])
        out.append(f"{indent}_check_frame_bounds(self)\n")
        i = line_end
    body = "".join(out)
    # Insert the helper's definition right after the last top-level import line.
    import_lines = list(re.finditer(r"^(?:from|import)\s.+$", body, re.MULTILINE))
    if import_lines:
        insert_at = import_lines[-1].end()
        body = body[:insert_at] + "\n" + _FRAME_GUARD_SOURCE + body[insert_at:]
    else:
        body = _FRAME_GUARD_SOURCE + body
    return body


def _already_clamped(expr: str) -> bool:
    """True if expr is already a max(...) call -- the SYSTEM_PROMPT's own example
    text ("clamp it instead: run_time=max(0.05, computed_value)") means the model
    sometimes writes this itself. Wrapping an already-wrapped expression in another
    max(0.05, ...) nests unbalanced parens and produces a SyntaxError (confirmed:
    this broke a real scene) -- skip it if it's already safe.
    """
    return expr.strip().startswith("max(")


def _clamp_zero_durations(code: str) -> str:
    """Manim hard-errors on a non-positive self.wait()/run_time -- and not just on
    a literal 0: a beat's sub-durations occasionally don't actually sum to its
    budget, leaving a computed expression like self.wait(20.66 - (14.96 + ...))
    that evaluates to a small negative number at render time. A regex that only
    catches literal zero misses that. Wrapping every occurrence in max(0.05, ...)
    instead makes the guarantee unconditional, regardless of how the model got
    the arithmetic wrong.
    """
    # self.wait(<arg>) -- always exactly one argument, so find its matching ')'.
    out = []
    i = 0
    marker = "self.wait("
    while True:
        idx = code.find(marker, i)
        if idx == -1:
            out.append(code[i:])
            break
        open_paren = idx + len(marker) - 1
        close_paren = _find_matching_paren(code, open_paren)
        if close_paren == -1:
            out.append(code[i:])
            break
        arg = code[open_paren + 1 : close_paren]
        out.append(code[i:idx])
        out.append(f"self.wait({arg})" if _already_clamped(arg) else f"self.wait(max(0.05, {arg}))")
        i = close_paren + 1
    code = "".join(out)

    # run_time=<expr> -- a kwarg inside a larger call; the expression ends at the
    # next top-level comma or closing bracket (bracket-depth-aware, since the
    # expression itself may contain nested parens).
    out = []
    i = 0
    marker = "run_time="
    while True:
        idx = code.find(marker, i)
        if idx == -1:
            out.append(code[i:])
            break
        start = idx + len(marker)
        depth = 0
        j = start
        while j < len(code):
            ch = code[j]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                if depth == 0:
                    break
                depth -= 1
            elif ch == "," and depth == 0:
                break
            j += 1
        expr = code[start:j]
        out.append(code[i:idx])
        out.append(f"run_time={expr}" if _already_clamped(expr) else f"run_time=max(0.05, {expr})")
        i = j
    return "".join(out)


def _last_code_block(code: str, class_name: str) -> str:
    """Despite "respond with ONLY code", the model occasionally writes a full
    scene, then a second corrected attempt after an internal ``` separator —
    strip_code_fences only strips a leading/trailing fence, so a fence left in
    the middle otherwise becomes a SyntaxError. Keep the last complete class
    definition, since a self-revision is the model's later, presumably-fixed
    attempt.
    """
    segments = [s for s in _FENCE_LINE.split(code) if f"class {class_name}(" in s]
    return segments[-1].strip() if segments else code


_CLASS_LINE = re.compile(r"^class\s+(\w+)\s*\(", re.MULTILINE)


def _last_class_definition(code: str, class_name: str) -> str:
    """Companion to _last_code_block for the same self-revision habit without a
    ``` fence in between: "let's discard and rebuild cleanly" followed by a
    second (or third) full `class SceneN(Scene):` definition, as plain valid
    Python. Nothing crashes -- Python just silently uses whichever definition
    executes last -- so this survives the lint/render loop untouched and only
    shows up as 2-3x redundant code and, worse, earlier abandoned attempts
    sitting in the file looking like they're part of what actually runs. Keep
    the header (imports, anything before the first class line) plus only the
    final definition.
    """
    matches = [m for m in _CLASS_LINE.finditer(code) if m.group(1) == class_name]
    if len(matches) <= 1:
        return code
    header = code[: matches[0].start()]
    return header + code[matches[-1].start() :]

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

When the brief involves showing or explaining Python code (a list vs. an array, a function \
call, a computed result) — the common case for a programming course — render it as real, \
syntax-highlighted code with Manim's `Code` mobject, never as plain `Text`. Construct it as:
    Code(code_string="...", language="python", formatter_style="monokai", \
background="window", add_line_numbers=True)
Use `formatter_style="monokai"` and `background="window"` for every code scene in this course \
so all code blocks look consistent across scenes generated independently of each other.
- A `Code` mobject's default size is driven entirely by its text — a long line (an import, a \
function call with several arguments, a comment) or many lines routinely renders wider or \
taller than the ~14×8-unit frame, with no automatic scaling. Every single `Code(...)` \
construction must be immediately followed by an explicit size cap before it is positioned — \
either chained directly (`Code(code_string="...", ...).scale_to_fit_width(10)`) or as the next \
line (`code_block.scale_to_fit_width(10)`). Use `.scale_to_fit_width(10)` for code with long \
lines, `.scale_to_fit_height(5)` for code with many lines, or both if it risks overflowing \
either dimension — never leave a `Code` block at its unscaled default size and assume it fits.
- To highlight a specific line as the narration reaches it, index the `Code` object's \
`.code_lines` attribute — a list of `VGroup`s, one per source line, `code_lines[0]` is the \
first line — with `Indicate`, a `SurroundingRectangle`, or `.animate.set_color(...)`, timed to \
the beat that names it. There is no `.code` or `.chars` attribute on `Code` — using either \
raises `AttributeError` and always crashes; `.code_lines` is the only way to reach individual \
lines.
- To show code changing (a list becoming an array, a line being added), build two `Code` \
objects and `Transform` or `TransformMatchingShapes` between them — don't edit text in place.
- When the brief calls for showing a result (array values, shape, dtype, printed output), pair \
the `Code` block with a small `Text` beneath it showing that output, rather than leaving the \
result implicit.

Layout, framing, and pacing — the most common failure modes in this pipeline; fix them \
explicitly rather than leaving them to chance:
- Overlap: never leave two mobjects occupying the same or intersecting screen space unless \
that overlap is the deliberate point of the shot. Position every new mobject deliberately \
relative to what's already on screen, using `.next_to()`, `.to_edge()`, `.arrange()`, or an \
explicit `.move_to()` — never rely on the default ORIGIN placement for a second object once \
something else is already there. When a piece of content is done being discussed and a new, \
unrelated piece takes over the same part of the screen, `FadeOut`/`Transform` the old one out \
before or as the new one comes in — don't just stack the new one on top of it. This matters for \
large elements (`Code`, `Table`, a multi-item `VGroup`) precisely because they're big enough to \
visually swallow whatever's underneath — but it matters just as much for a scene with many \
small `Text` labels appearing one after another (a word or phrase called out beat by beat): each \
new label needs the previous one(s) faded out first, or they silently accumulate into a pile of \
overlapping text. Track every label you create in a variable and explicitly `FadeOut` it before \
or as the next one appears — never leave a label on screen just because nothing told you to \
remove it. Before writing a beat's code, mentally list everything still visible from the \
previous beat and confirm each item is either still relevant or explicitly faded.
- Off-screen: the visible frame is about 14 units wide and 8 units tall, centered on the \
origin — stay within roughly ±6.5 horizontally and ±3.5 vertically, and scale down anything \
that would exceed it. Before placing a `Text`, `Code` block, or `VGroup`, consider its actual \
rendered size — long text, wide tables, and multi-element groups need `.scale()` or \
`.scale_to_fit_width()` / `.scale_to_fit_height()` to fit the frame, not a guess at font size. \
When arranging several elements with `.arrange()`, check the combined width/height against the \
frame before finalizing positions, and scale the whole group down if it would overflow. Prefer \
a few large, clearly readable elements over many small, crowded ones.
- Stale copies: never build a `SurroundingRectangle`, highlight box, or `.next_to()`-positioned \
label from a `.copy()` of a mobject that the *original* is later moved or scaled via `.animate` \
— the copy stays frozen at its pre-animation position, so the highlight/label built from it \
visually drifts away from the real, now-moved content and ends up overlapping whatever's on \
screen at the copy's old location instead. If a highlight or label needs to track a mobject \
through a zoom/move, build it AFTER that mobject's own move/scale animation has already played, \
from the live mobject at its current position — never from an earlier copy.
- Re-zooming an already-fitted group: before scaling up an existing group for a "zoom" effect \
(`.animate.scale(k)` with k > 1), check whether that group was already sized close to the frame \
with `.scale_to_fit_width()`/`.scale_to_fit_height()` — scaling a group that already fills most \
of the ~14-unit-wide frame will push its edges off-screen. To zoom in on part of a wide \
structure (one column of a table, one region of a diagram), scale and reposition just that \
subset, never the whole already-fitted group; if you do scale the whole group, re-check its new \
width/height against the frame afterward.
- Labels near multi-part structures: when placing a new label near a structure that has several \
parts stacked closely together (a table's header row directly above its data row, a title above \
a subtitle), position the label relative to the structure's full bounding box — \
`.next_to(whole_group, DOWN, buff=...)` — not relative to just one sub-piece of it \
(`.next_to(header_row, DOWN, buff=...)`), since a buff small enough to sit close under the \
sub-piece can still land the label on top of the next sub-piece immediately below it.
- Pacing: the brief gives exact timestamps per beat — treat these as a real time budget, not a \
suggestion. For each beat, compute its duration as (this beat's timestamp − the previous \
beat's timestamp) and give the corresponding `self.play(..., run_time=<that duration>)` or \
`self.wait(<that duration>)` exactly that duration — never leave an animation at Manim's \
default 1-second `run_time` when the beat calls for something else, and never let `self.wait()` \
gaps silently accumulate drift. The sum of every beat's duration must equal the target \
duration, and each beat's on-screen content must actually be showing during that beat's \
timestamp window, not early or late.
- Loops are the single most common way this budget gets blown — watch for them specifically. \
When a beat's action repeats over several items (a `for` loop over columns, neurons, list \
elements, anything), the beat's duration is the TOTAL for every iteration combined, not the \
duration of each iteration. Divide first: if a beat spans 4.0 seconds and the loop has 5 \
items, each iteration gets 4.0/5 = 0.8 seconds total — split across whatever `self.play()` and \
`self.wait()` calls happen inside that one iteration (e.g. `run_time=0.6` for the play plus \
`self.wait(0.2)`), not 4.0 seconds each. Concretely, `for item in items: self.play(..., \
run_time=4.0)` inside a beat is always wrong — compute `per_item = beat_duration / len(items)` \
before the loop and use fractions of `per_item` inside it. After writing a loop, mentally sum \
`per_item * len(items)` and confirm it equals the beat's duration, not a multiple of it.

Code size — keep it tight, both for your own token budget and because repeated, copy-pasted \
code is exactly where mistakes hide (a forgotten `FadeOut`, a mismatched position, a label \
left on screen):
- Prefer a `for` loop or list/dict comprehension over writing near-identical code for each \
item of a repeated structure (each cell of a grid, each column of a table, each entry in a \
list) — never manually write out N copies of almost-the-same code when a loop over `range(N)` \
or the data itself produces the same result. A scene with a large grid or many similar items \
should still be a short loop, not hundreds of lines of repetition.
- Use Manim's own grouping/arrangement helpers (`VGroup(*[...])`, `.arrange()`, \
`.arrange_in_grid()`) to build and position repeated elements together, rather than computing \
and setting each one's position by hand.
- If a scene's natural content genuinely requires many distinct elements (a large data table, \
many labeled items), favor showing a representative subset or a summarized view over building \
every single element individually — the narration rarely needs every cell of a 30-item grid \
animated in individually to make its point.

Visual consistency across scenes — every scene in this course is generated independently, so \
these choices must be made the same way every time rather than left to each generation's \
discretion:
- Font: use Manim's default font (don't set a custom `font=` on `Text`) so every scene renders \
text identically.
- Color palette: use only `WHITE` for primary text/labels, `YELLOW` for highlights/emphasis, \
`GREY`/`GREY_B` for secondary/de-emphasized text, `RED`/`GREEN`/`BLUE` for data or comparison \
categories, and `monokai`-scheme colors (handled automatically by `Code`'s \
`formatter_style="monokai"`) for code. Don't introduce other named colors or hex values — \
sticking to this small palette is what makes independently-generated scenes look like one \
consistent course instead of a visibly different style each time.

Rules:
- Import only from `manim` (`from manim import *`) plus the Python standard library and numpy.
- The class must be named exactly `{class_name}`.
- Total animation runtime must be within 1 second of the target duration. Use self.wait() to \
fill gaps.
- No file I/O, no network access, no external assets — concretely, this means never use \
`ImageMobject` or `SVGMobject`: both load a file from disk, and no images or SVGs are supplied \
to this pipeline, so both always crash. Build every visual from Manim's own vector primitives \
instead — `Rectangle`, `Square`, `Circle`, `Ellipse`, `Line`, `Dot`, `Polygon`, `Arc`, `VGroup` \
— composed together. A striped image is a `VGroup` of alternating-color thin `Rectangle`s, not \
a hand-built pixel array; a mouse/animal is a `Circle`/`Ellipse` body with smaller shapes for \
ears/features, not an SVG; an icon is built from primitives, not loaded.
- If a specific figure the brief calls for genuinely cannot be built well from Manim's \
primitives (a realistic anatomical drawing, a photorealistic object), do not force a bad or \
misleading substitute — omit that figure and cover the same point with what the scene already \
has (the narration, code, text, or a simpler shape) rather than adding a placeholder block, a \
generic icon that misrepresents the thing, or an unrelated stand-in.
- When a beat describes a visual only abstractly or by example (e.g. "a ledger or a list of \
numbers being adjusted by hand", "some kind of chart") without giving the actual values, \
labels, or text it should show, do not invent specific-looking data to fill it in — no made-up \
timestamps, names, or numbers that aren't in the brief, narration, or the values already used \
elsewhere in this same scene. Fabricated specifics read as authoritative but are simply false. \
Render the visual in a way that stays honestly generic instead: unlabeled rows or blank lines, \
placeholder dashes, or ellipses — or reuse a real value that already appears in this scene's \
brief/narration if one fits — rather than a number, name, or date that was never given to you.
- Every `run_time=` on `self.play(...)` and every duration passed to `self.wait(...)` must be a \
positive number greater than 0 — Manim raises a hard error on 0 or negative, for both calls. \
This applies just as much to `self.wait()` calls used to pad out to an exact timestamp (e.g. \
`self.wait(0.0)  # fill to 5.44`) as it does to computed per-item loop durations — if the gap \
being padded is 0 or would round to ~0, skip the `self.wait()` call entirely rather than call it \
with 0; never emit a literal `self.wait(0.0)` or `run_time=0`. For a divided-up per-item duration \
(see the Pacing rule above) that could round down to 0 for a very short beat with many items, \
clamp it instead: `run_time=max(0.05, computed_value)`. For something that should appear \
instantly with no animation at all, use `self.add(...)` instead of `self.play(..., run_time=0)`.
- `.move_to()`, `.shift()`, and every other positioning method take a single point argument — \
either `np.array([x, y, z])` or a combination of directional vectors like `RIGHT*x + UP*y` — \
never separate x, y, z arguments. `mob.move_to(x, y, z)` raises `TypeError: 'float' object is \
not subscriptable`; for a grid or computed layout, build the point explicitly first: \
`mob.move_to(np.array([x, y, 0]))`.
- Never reference `self.camera.frame` — that attribute only exists on `MovingCameraScene`, and \
every scene here subclasses plain `Scene`, so it always crashes with an `AttributeError`. For a \
"zoom in/out" or "pan" effect, animate the mobjects themselves instead — `group.animate.scale(...)`, \
`.move_to(...)`, or `.shift(...)` on the `VGroup`/mobject in question — never the camera.
- Never use `MathTex` or `Tex` — both shell out to a LaTeX installation that isn't present in \
this environment, so both always crash. For equations and math notation, use plain `Text` \
instead, built with unicode directly (e.g. "ẋ = μx − ωy", "σ²", "x₁", "≈", "√") — Manim's \
default font renders these fine without LaTeX.
- Respond with ONLY the Python code. No markdown fences, no commentary.
- If you're given the previous scene's code below, this scene plays immediately after it in the \
final video — when the brief refers to the same data, object, or example the previous scene \
used, reuse its exact values (the same numbers, labels, variable names), not new ones you make \
up. Two consecutive scenes about the same dataset must show identical numbers, not two \
different-looking invented versions of it.

{manim_api_reference}
"""


def generate_code(
    class_name: str,
    render_prompt: str,
    duration: float,
    previous_code: Optional[str] = None,
    feedback: Optional[str] = None,
    previous_scene_code: Optional[str] = None,
) -> str:
    system = SYSTEM_PROMPT.format(class_name=class_name, manim_api_reference=MANIM_API_REFERENCE)
    user_prompt = f"Animation brief:\n{render_prompt}\n\nTarget duration: {duration:.2f} seconds"
    if previous_scene_code:
        user_prompt += (
            f"\n\nThe previous scene's code (for continuity — reuse its exact values where the "
            f"brief refers to the same data/object, see the continuity rule above):\n{previous_scene_code}"
        )
    if previous_code and feedback:
        user_prompt += (
            f"\n\nYour previous attempt had this problem:\n{feedback}\n\n"
            f"Previous code:\n{previous_code}\n\nFix it."
        )
    # 16384, not 8192 -- confirmed a real scene (589+ lines, a large repeated-grid
    # visualization) still hadn't finished at 8192 and got cut off mid-expression,
    # producing a SyntaxError. DeepSeek has plenty of headroom for this; the local
    # Qwen server's 16384-token *shared* context budget doesn't apply here since
    # this is a completion-only (not prompt+completion) ceiling on a separate endpoint.
    code = strip_code_fences(generate_code_model(system=system, prompt=user_prompt, max_tokens=16384))
    code = _last_code_block(code, class_name)
    code = _last_class_definition(code, class_name)
    code = _clamp_zero_durations(code)
    return _inject_frame_guard(code)
