"""Composite asset builder: combines registered SVG icons with matplotlib layout
(positioning, arrows, labels, a consistent color palette) into a single flat PNG,
then registers the result back into the same asset library that ASSET_PATHS pulls
from in generated Manim code.

Why this exists: a beat that's really a small infographic -- several icons, arrows,
and labels in one fixed layout (a funnel diagram, a mind map) -- looks crude and
inconsistent if an LLM re-lays it out live from Manim primitives on every render
(CONTENT_GUIDELINES.md calls this out: hand-drawn recognizable objects read as
"stickman" art). Designing it once here as a flat image, using real open-licensed
icons plus matplotlib's text/arrow rendering, gives full design control and looks
identical every time the scene references it. Simple single-icon references
(a heart, a brain, a stopwatch) don't need this -- use ASSET_PATHS[...] with
SVGMobject directly in Manim as before; this module is for compound layouts only.

Run this file directly to (re)build every composite asset this project's scenes
reference. Import `icon_array`/`place_icon` to add new ones without duplicating
the SVG-rasterization plumbing.
"""

import io
from pathlib import Path

import cairosvg
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

from utils.asset_library import get_asset, register_asset

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "generated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# One small palette, reused across every composite so they read as one visual
# system rather than five separately-designed images.
INK = "#1b1f24"
PAPER = "#faf9f5"
MUTED = "#c9c6bd"
PILLAR_COLOR = {
    "representation": "#5b6ee1",  # Linear Algebra
    "change": "#e0763f",  # Calculus & Dynamical Systems
    "uncertainty": "#3fa66e",  # Statistics & Probability
    "signals": "#c74e6d",  # Signals & Frequency
    "learning": "#8a5bc9",  # Learning & Optimization
}


def icon_array(key: str, px: int = 256, color: str | None = None) -> np.ndarray:
    """Rasterizes a registered SVG icon to an RGBA array, optionally recoloring its
    fill. Bootstrap Icons ship as a single `fill="currentColor"` path, so a plain
    text substitution is enough to retint -- no per-color source files needed."""
    path = get_asset(key)
    if path is None:
        raise KeyError(f"{key!r} is not a registered asset -- register it first (see register_icons.py).")
    svg_text = path.read_text(encoding="utf-8")
    if color is not None:
        svg_text = svg_text.replace('fill="currentColor"', f'fill="{color}"', 1)
    png_bytes = cairosvg.svg2png(bytestring=svg_text.encode("utf-8"), output_width=px, output_height=px)
    return mpimg.imread(io.BytesIO(png_bytes))


def place_icon(ax, key: str, xy: tuple[float, float], zoom: float = 0.18, color: str | None = None):
    """Places a registered icon on a matplotlib Axes at data coordinates `xy`."""
    im = OffsetImage(icon_array(key, color=color), zoom=zoom)
    ab = AnnotationBbox(im, xy, frameon=False)
    ax.add_artist(ab)
    return ab


def _new_axes(w: float = 10, h: float = 6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def _save_and_register(fig, key: str, description: str, out_name: str) -> Path:
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    dest = register_asset(key, out_path, "image", description)
    print(f"built + registered {key!r} -> {dest}")
    return dest


def build_measurement_funnel():
    """Part 1's funnel: EEG / MRI / reaction-time measurements funneling into
    'MATHEMATICS', opening into a lightbulb ('pattern -> insight -> decision')."""
    fig, ax = _new_axes(11, 6.5)

    sources = [
        ("clipboard_pulse", "EEG\n1,000,000+ voltage values", 1.6),
        ("hospital", "MRI\n500,000+ voxels", 5.5),
        ("stopwatch", "Reaction time\none number per trial", 9.4),
    ]
    for key, label, x in sources:
        place_icon(ax, key, (x, 5.6), zoom=0.26, color=INK)
        ax.text(x, 4.65, label, ha="center", va="top", fontsize=10, color=INK, linespacing=1.4)
        ax.add_patch(FancyArrowPatch((x, 4.15), (5.5, 3.15), connectionstyle="arc3,rad=0.0",
                                      arrowstyle="-|>", mutation_scale=14, color=MUTED, linewidth=1.4))

    funnel = Polygon([(3.7, 3.3), (7.3, 3.3), (6.0, 1.9), (5.0, 1.9)],
                      closed=True, facecolor="#eceadf", edgecolor=INK, linewidth=1.6)
    ax.add_patch(funnel)
    ax.text(5.5, 2.75, "MATHEMATICS", ha="center", va="center", fontsize=13,
            fontweight="bold", color=INK)

    ax.add_patch(FancyArrowPatch((5.5, 1.9), (5.5, 1.25), arrowstyle="-|>",
                                  mutation_scale=14, color=INK, linewidth=1.6))
    place_icon(ax, "lightbulb", (5.5, 0.65), zoom=0.3, color="#c98a1f")
    ax.text(5.5, -0.05, "pattern  ->  insight  ->  decision", ha="center", va="top",
            fontsize=10.5, color=INK)
    ax.set_ylim(-0.5, 6.5)

    _save_and_register(
        fig,
        "fig_measurement_funnel",
        "Composite PNG (matplotlib layout + Bootstrap Icons, MIT licensed) -- the "
        "'measurements funnel into mathematics' diagram for Part 1 (why neuroscience "
        "needs mathematics). Use via ImageMobject(ASSET_PATHS['fig_measurement_funnel']).",
        "fig_measurement_funnel.png",
    )


_NODES = [
    # key, label, icon, angle (degrees, 0 = due right, ccw), tier
    ("representation", "Linear Algebra\nRepresentation", "grid_matrix", 90),
    ("change", "Calculus &\nDynamical Systems\nChange", "graph_trend", 162),
    ("uncertainty", "Statistics &\nProbability\nUncertainty", "bar_chart", 234),
    ("signals", "Signals &\nFrequency", "soundwave", 306),
    ("learning", "Learning &\nOptimization", "cpu", 18),
]


def build_mind_map(active: set[str], key: str, description: str, out_name: str):
    """The recurring 5-node mind map. `active` selects which nodes render in full
    color (already introduced) vs muted gray (not yet introduced) -- call this with
    a different `active` set to produce each progressive-reveal state the course
    needs, without redesigning the layout each time."""
    fig, ax = _new_axes(9, 9)
    cx, cy, r = 4.5, 4.5, 3.2

    for key_, label, icon, angle in _NODES:
        rad = np.radians(angle)
        x, y = cx + r * np.cos(rad), cy + r * np.sin(rad)
        is_on = key_ in active
        color = PILLAR_COLOR[key_] if is_on else MUTED
        style = "-" if is_on else "--"
        ax.add_patch(FancyArrowPatch((cx, cy), (x, y), connectionstyle="arc3,rad=0.0",
                                      arrowstyle="-", linestyle=style, linewidth=1.6,
                                      color=color, alpha=0.9 if is_on else 0.6,
                                      shrinkA=38, shrinkB=30))
        ax.add_patch(Circle((x, y), 0.62, facecolor="white", edgecolor=color, linewidth=2.2, zorder=3))
        place_icon(ax, icon, (x, y), zoom=0.26, color=color if is_on else MUTED)
        ax.text(x, y - 0.95, label, ha="center", va="top", fontsize=9.5,
                color=INK if is_on else MUTED, linespacing=1.3, zorder=3)

    ax.add_patch(Circle((cx, cy), 0.95, facecolor=INK, edgecolor=INK, zorder=4))
    ax.text(cx, cy, "MATHEMATICS\nIN\nNEUROSCIENCE", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=PAPER, linespacing=1.3, zorder=5)

    ax.set_xlim(-0.3, 9.3)
    ax.set_ylim(-0.9, 9.3)
    _save_and_register(fig, key, description, out_name)


def build_all():
    build_measurement_funnel()
    build_mind_map(
        active={"representation", "change", "uncertainty"},
        key="fig_mindmap_three_pillars",
        description=(
            "Composite PNG (matplotlib layout + Bootstrap Icons, MIT licensed) -- the "
            "mind map with only the three foundational pillars (Linear Algebra, Calculus "
            "& Dynamical Systems, Statistics & Probability) lit, the two applied nodes "
            "still muted. Use for Part 3's transition and any pillar-recap beat via "
            "ImageMobject(ASSET_PATHS['fig_mindmap_three_pillars'])."
        ),
        out_name="fig_mindmap_three_pillars.png",
    )
    build_mind_map(
        active={"representation", "change", "uncertainty", "signals", "learning"},
        key="fig_mindmap_full",
        description=(
            "Composite PNG (matplotlib layout + Bootstrap Icons, MIT licensed) -- the "
            "complete 5-node mind map, all nodes lit. Use for Part 7's expansion and the "
            "closing full reveal via ImageMobject(ASSET_PATHS['fig_mindmap_full'])."
        ),
        out_name="fig_mindmap_full.png",
    )


if __name__ == "__main__":
    build_all()
