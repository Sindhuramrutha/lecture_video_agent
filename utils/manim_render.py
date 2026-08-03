"""Deterministic Manim render wrapper. No LLM involvement — pure execution.

Renders at low quality (480p15) for fast iteration through the critique
loop. Swap to a higher -q flag for a final pass once scenes are approved.
"""

import subprocess
import sys
from pathlib import Path


class RenderError(Exception):
    pass


def render_scene(code: str, class_name: str, work_dir: Path) -> Path:
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    code_path = work_dir / f"{class_name}.py"
    code_path.write_text(code)

    media_dir = work_dir / "media"
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "manim", "render", "-ql",
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
        raise RenderError(result.stderr[-4000:])

    matches = list(media_dir.glob(f"videos/{class_name}/*/{class_name}.mp4"))
    if not matches:
        raise RenderError(f"Manim exited cleanly but produced no output for {class_name}")
    return matches[0]
