"""Deterministic Manim render wrapper. No LLM involvement — pure execution.

Renders at medium quality (720p30) -- the floor for published content;
never drop below this. -qh (1080p60) is available if a sharper final pass
is ever needed, at real render-time cost -- 720p was chosen specifically
to stay at the minimum that still satisfies that floor, not higher.
"""

import subprocess
import sys
from pathlib import Path

from utils.render_errors import RenderError

__all__ = ["RenderError", "render_scene"]


def _clean_terminal_output(text: str) -> str:
    """Collapses carriage-return-redrawn lines (Manim/rich rewrite the same
    progress-bar line dozens of times via \\r) down to their final state, the
    way a real terminal would display them -- otherwise a captured stderr on
    failure is dominated by intermediate per-animation progress redraws,
    burying the actual traceback under a wall of noise.
    """
    lines = [raw_line.rsplit("\r", 1)[-1] for raw_line in text.split("\n")]
    # Collapse the runs of blank lines progress-bar redraws tend to leave behind.
    cleaned, blank_run = [], 0
    for line in lines:
        if line.strip():
            blank_run = 0
            cleaned.append(line)
        else:
            blank_run += 1
            if blank_run <= 1:
                cleaned.append(line)
    return "\n".join(cleaned)


def render_scene(code: str, class_name: str, work_dir: Path) -> Path:
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    code_path = work_dir / f"{class_name}.py"
    code_path.write_text(code, encoding="utf-8")

    media_dir = work_dir / "media"
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "manim", "render", "-qm", "--progress_bar", "none",
                "--media_dir", str(media_dir), str(code_path), class_name,
            ],
            capture_output=True,
            text=True,
            cwd=work_dir,
            timeout=180,
        )
    except subprocess.TimeoutExpired as exc:
        raise RenderError(f"Render timed out after 180s for {class_name}") from exc
    if result.returncode != 0:
        raise RenderError(_clean_terminal_output(result.stderr)[-4000:])

    matches = list(media_dir.glob(f"videos/{class_name}/*/{class_name}.mp4"))
    if not matches:
        raise RenderError(f"Manim exited cleanly but produced no output for {class_name}")
    return matches[0]
