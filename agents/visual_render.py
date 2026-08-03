"""Visual Render stage: prompt -> code -> render, per scene. No automated
critique — a vision-critique loop was burning several API calls per
scene to make a judgment a human was going to re-check anyway. Instead:
render once, a human looks at it, and optionally reprompts with free
text (see `instruction` below) — as many rounds as the human wants.

The small retry cap here (MAX_RENDER_ATTEMPTS) only covers genuine
technical failures (the generated code crashed / produced no output),
not aesthetic judgment — fixing a crash automatically is cheap and
necessary; judging quality is the human's call.

Dispatches each scene to its chosen renderer (manim or matplotlib).
Each `_render_with_*` function doubles as both the initial-render path
(instruction=None) and the manual-revise path (instruction=<human
text>, continuing from the scene's existing code) — same function,
called from a different place in the notebook. image_only and
audio_only scenes pass through untouched; those pipelines don't exist
yet. Adding a new renderer (e.g. Motion Canvas) means writing one more
`_render_with_*` function and registering it in RENDERERS.
"""

from pathlib import Path
from typing import Callable, Optional

from tqdm.auto import tqdm

from agents import matplotlib_coder, scene_coder
from agents.prompt_maker import make_render_prompt
from state import Scene, VideoState
from utils import manim_render, matplotlib_render
from utils.ffmpeg_utils import duration_seconds
from utils.notify import notify
from utils.storage import project_dir

MAX_RENDER_ATTEMPTS = 2  # technical-failure retries only, not aesthetic judgment
DURATION_SHORTFALL_TOLERANCE = 2.0  # seconds; below this, treat as a real bug, not rounding

CodeFn = Callable[[Optional[str], Optional[str]], str]
RenderFn = Callable[[str], Path]


def _run_render_loop(
    code_fn: CodeFn,
    render_fn: RenderFn,
    initial_code: Optional[str],
    initial_feedback: Optional[str],
    target_duration: float,
):
    code = initial_code
    feedback = initial_feedback
    video_path = None
    succeeded = False
    attempts = 0

    for attempt in range(1, MAX_RENDER_ATTEMPTS + 1):
        attempts = attempt
        code = code_fn(code, feedback)
        if not code.strip():
            feedback = "The previous attempt returned no code at all. Write the full scene."
            continue
        try:
            candidate_path = render_fn(code)
        except (manim_render.RenderError, matplotlib_render.RenderError) as exc:
            feedback = f"Render failed: {exc}"
            continue

        actual_duration = duration_seconds(candidate_path)
        shortfall = target_duration - actual_duration
        if shortfall > DURATION_SHORTFALL_TOLERANCE:
            feedback = (
                f"The rendered clip is only {actual_duration:.1f}s long but needs to reach "
                f"{target_duration:.1f}s to cover the full narration — the animation stops "
                f"too early. Cover more of the brief, or add self.wait() time, to fill the "
                f"remaining {shortfall:.1f}s."
            )
            continue

        video_path = candidate_path
        succeeded = True
        feedback = None
        break

    return code, video_path, attempts, succeeded, feedback


def _render_with_manim(scene: Scene, project_root: Path, instruction: Optional[str] = None) -> Scene:
    render_prompt = scene.render_prompt or make_render_prompt(scene)
    duration = duration_seconds(Path(scene.audio_path))
    class_name = f"Scene{scene.id}"
    work_dir = project_root / "manim" / class_name

    code, video_path, attempts, succeeded, feedback = _run_render_loop(
        code_fn=lambda prev, fb: scene_coder.generate_code(class_name, render_prompt, duration, prev, fb),
        render_fn=lambda code: manim_render.render_scene(code, class_name, work_dir),
        initial_code=scene.code if instruction else None,
        initial_feedback=instruction,
        target_duration=duration,
    )
    return scene.model_copy(
        update={
            "render_prompt": render_prompt,
            "code": code if code else scene.code,
            "video_path": str(video_path) if video_path else scene.video_path,
            "render_attempts": scene.render_attempts + attempts,
            "render_status": "rendered" if succeeded else "failed",
            "last_feedback": feedback if not succeeded else None,
        }
    )


def _render_with_matplotlib(scene: Scene, project_root: Path, instruction: Optional[str] = None) -> Scene:
    render_prompt = scene.render_prompt or make_render_prompt(scene)
    duration = duration_seconds(Path(scene.audio_path))
    script_name = f"scene_{scene.id}"
    work_dir = project_root / "matplotlib" / script_name

    code, video_path, attempts, succeeded, feedback = _run_render_loop(
        code_fn=lambda prev, fb: matplotlib_coder.generate_code(render_prompt, duration, prev, fb),
        render_fn=lambda code: matplotlib_render.render_scene(code, script_name, work_dir),
        initial_code=scene.code if instruction else None,
        initial_feedback=instruction,
        target_duration=duration,
    )
    return scene.model_copy(
        update={
            "render_prompt": render_prompt,
            "code": code if code else scene.code,
            "video_path": str(video_path) if video_path else scene.video_path,
            "render_attempts": scene.render_attempts + attempts,
            "render_status": "rendered" if succeeded else "failed",
            "last_feedback": feedback if not succeeded else None,
        }
    )


RENDERERS: dict[str, Callable[..., Scene]] = {
    "manim": _render_with_manim,
    "matplotlib": _render_with_matplotlib,
}


def visual_render_node(state: VideoState) -> dict:
    """Batch path: renders every scene once, no human review in between.

    Prefer the notebook's interactive per-scene loop when you want to
    review/revise each clip before moving on — this is for when you
    just want everything rendered and plan to review at the end.
    """
    project_root = project_dir(state.topic)
    updated_scenes = []
    for scene in tqdm(state.scenes, desc="Visual Render"):
        render = RENDERERS.get(scene.scene_type)
        updated_scenes.append(render(scene, project_root) if render else scene)
    notify(f"Visual Render done — {len(updated_scenes)} scene(s)")
    return {"scenes": updated_scenes}
