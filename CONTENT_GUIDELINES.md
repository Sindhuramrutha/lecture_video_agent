# Writing detailed content for the video pipeline

This is the one input you author by hand — everything downstream (scene
script, animation briefs, Manim code, narration audio) is generated from it.
The pipeline is built to be faithful to what you write, not to embellish or
compress it (see `agents/scene_script.py`'s fidelity rules), so the quality
and completeness of this document directly determines the quality of the
video. There is no asset library — no external images, SVGs, or diagrams get
pulled in — so everything on screen either comes from Manim's own vector
primitives, drawn according to what you describe, or is left out. Nothing you
reference by filename ("see the diagram below") will appear.

## Format

- Supported file types: `.txt`, `.md`, `.pdf`, `.docx` (see
  `utils/references.py`).
- Write it as textbook prose: sections and subsections, worked examples,
  equations, explanations of *why* something is true — not bullet-point
  notes or a slide outline. The scene script agent narrates what's actually
  written, in full depth; thin, outline-style notes produce thin, short
  scenes rather than getting padded out.
- Structure it the way you'd want it taught: the order you write subtopics
  in is the order they'll be narrated, scene by scene, with no reordering or
  gap-filling.

## Setting the target duration

Two ways, and they combine:

1. Pass `duration_minutes` on `VideoState` — an approximate total across all
   scenes combined (not per-scene).
2. State a target duration directly inside the content itself (e.g. a line
   like "Target duration: 15 minutes" near the top of a section). If present,
   this is treated as authoritative alongside the state-level value.

Either way, the target is a ceiling for trimming, not a floor to pad to: if a
full, uncompressed narration of what you wrote comes in under the target,
that's left as-is — a short video from short source material is correct, not
a bug. If it would run over, only then does the model trim (redundant
restatement, secondary asides first — never a core fact, equation, or
example). If you want a video of a specific length, write source material
that actually takes that long to explain in full; the target duration alone
won't stretch thin content to fit it.

## Equations and math notation

Write equations as ordinary readable text (`x' = μx − ωy`, `σ²`, `∫f(x)dx`)
rather than dense LaTeX markup (`\frac{}{}`, `\int_{a}^{b}`). The renderer
(`agents/scene_coder.py`) never uses `MathTex`/`Tex` — no LaTeX is installed
in this environment — so on-screen math is always plain Unicode text.
Heavy LaTeX in the source doesn't render better; it just makes the
scene-script step's job of narrating and re-expressing it harder.

## Code examples (programming content)

Include real, runnable code snippets directly in the content, not
descriptions of code ("a function that loops over the list"). Real snippets
get rendered as actual syntax-highlighted `Code` blocks; described-but-absent
code has nothing concrete for the animation to show. If a snippet's output
matters (a printed value, an array's shape), state that output explicitly
too — it gets shown alongside the code.

## Tables and data

Numeric tables in the source get turned into an on-screen `Table` mobject.
Keep them to a size that's actually readable on a 14×8-unit frame — a handful
of columns and a few rows works well; a wide table with many columns will
either get uncomfortably shrunk or produce a cramped scene. If you have a
large dataset, describe the relevant slice or summary rather than pasting an
entire table.

## Figures and physical systems

- If a concept has a physical embodiment (a mass on a spring, a pendulum, a
  wave on a string, a circuit), describe the literal physical picture, not
  just the equation — Manim builds these from primitives (lines, circles,
  arcs) and shows the concrete object before the abstract math.
- If a figure genuinely can't be drawn well with basic vector shapes (a
  realistic anatomical drawing, a photograph), don't rely on it — the
  pipeline omits figures it can't build honestly rather than faking a
  placeholder. Cover the same point in words, code, or a simpler diagram
  instead.
- Don't reference external images/diagrams by name or "see below" — there's
  no mechanism to attach them. Describe what should appear in words.

## Ideal chapter size

**Aim for 800–1,500 words of detailed content per chapter (per file in
`input_files/`).** That's not a hard limit the pipeline enforces — it's the
range that keeps a chapter practical to run and review in one sitting, based
on the actual numbers from the real test case this pipeline was built and
hardened against: `pod1_numpy.txt` (1,008 words) produced 6 scenes and a
3:26 final video, start to finish.

Two things scale directly with word count, and both get unwieldy past this
range:
- **Scene count** — roughly 150–170 words of source content becomes one
  scene. 800–1,500 words lands you at 5–9 scenes, which is a comfortable
  number of silent clips to watch in one review pass. Push much past that
  and the one-halt review step turns into reviewing 20-30 clips at once.
- **`prepare` runtime** — each scene needs its own animation-brief call
  (Qwen, ~1-2 minutes) plus a render loop (DeepSeek + Manim, up to a couple
  of attempts each). More scenes means a longer unattended `--stage prepare`
  run before you have anything to look at.

If your source material is naturally longer than that (like the full
`python_course_3_numpy.docx`, ~6,500 words across 6 "Pods"), split it at its
own natural boundaries — one input file per Pod/section, one `run_chapter.py`
call per file — rather than feeding the whole thing in as a single chapter.

## Section sizing (within a chapter)

Scenes run roughly 15–45 seconds of narration each (`agents/scene_script.py`).
A subtopic that would take much longer than that to narrate in full will get
split into multiple scenes automatically; you don't need to pre-chunk it
yourself, just write it as a coherent, complete section.

## Reference example

`other_resources/python_course_3_numpy.docx` ("Pod 1") is the real content
used to test and harden this pipeline — worth skimming as a length/depth
reference for a single ~6-scene section.
