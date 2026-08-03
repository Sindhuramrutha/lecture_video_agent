"""Deterministic Matplotlib render wrapper. No LLM involvement — pure execution.

The generated script is trusted to save its own output as `output.mp4`
in its working directory (see agents/matplotlib_coder.py's contract).
"""

import shutil
import subprocess
import sys
from pathlib import Path


class RenderError(Exception):
    pass


def render_scene(code: str, script_name: str, work_dir: Path) -> Path:
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    code_path = work_dir / f"{script_name}.py"
    code_path.write_text(code)

    output_path = work_dir / "output.mp4"
    output_path.unlink(missing_ok=True)

    try:
        result = subprocess.run(
            [sys.executable, str(code_path)],
            capture_output=True,
            text=True,
            cwd=work_dir,
            timeout=180,
        )
    except subprocess.TimeoutExpired as exc:
        raise RenderError(f"Render timed out after 180s for {script_name}") from exc
    if result.returncode != 0:
        raise RenderError(result.stderr[-4000:])
    if not output_path.exists():
        raise RenderError("Script exited cleanly but did not produce output.mp4")

    final_path = work_dir / f"{script_name}.mp4"
    shutil.move(str(output_path), str(final_path))
    return final_path
