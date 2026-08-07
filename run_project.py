#!/usr/bin/env python
"""Run Manim generation/render/compose directly against a project folder,
without going through the LangGraph workflow or an LLM-mediated session.

Operates on the per-course folder layout under projects/<slug>/:
  prompts/scene_NN_prompt.txt   -- animation brief (must already exist)
  audio/scene_NN.wav            -- narration audio (must already exist, sets target duration)
  manim_code/scene_NN_SceneNN.py -- generated source (written by the `code` stage)
  rendered_video/scene_NN_with_audio.mp4 -- final muxed clip (written by `compose`)

Examples:
  python run_project.py --project python-course --scenes 1,3 --stage all
  python run_project.py --project python-course --stage render --scenes 1
  python run_project.py --project python-course --stage compose
  python run_project.py --project python-course            # all scenes, all stages

Each scene goes through the same lint -> render -> duration-check retry loop
as the LangGraph pipeline (see agents/visual_render.py's _run_render_loop),
so a scene that hits a known bug class or renders short gets fixed
automatically within MAX_ATTEMPTS before this script gives up on it.
"""

import argparse
import re
import sys
from pathlib import Path

from agents import scene_coder
from utils import av_compose, manim_lint, manim_render
from utils.ffmpeg_utils import duration_seconds

PROJECTS_DIR = Path(__file__).resolve().parent / "projects"
MAX_ATTEMPTS = 3
DURATION_SHORTFALL_TOLERANCE = 2.0  # seconds


def discover_scenes(project_dir: Path) -> list[int]:
    ids = []
    for p in sorted((project_dir / "prompts").glob("scene_*_prompt.txt")):
        m = re.match(r"scene_(\d+)_prompt\.txt", p.name)
        if m:
            ids.append(int(m.group(1)))
    return ids


def scene_paths(project_dir: Path, scene_id: int) -> dict:
    padded = f"{scene_id:02d}"
    class_name = f"Scene{scene_id}"
    return {
        "class_name": class_name,
        "brief": project_dir / "prompts" / f"scene_{padded}_prompt.txt",
        "audio": project_dir / "audio" / f"scene_{padded}.wav",
        "code": project_dir / "manim_code" / f"scene_{padded}_{class_name}.py",
        "work_dir": project_dir / "manim" / class_name,
        "final": project_dir / "rendered_video" / f"scene_{padded}_with_audio.mp4",
    }


def run_code_stage(project_dir: Path, scene_id: int) -> str:
    """Generate code, retrying on lint failures only (no render, fast iteration)."""
    paths = scene_paths(project_dir, scene_id)
    brief = paths["brief"].read_text()
    duration = duration_seconds(paths["audio"])

    code, feedback = None, None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        code = scene_coder.generate_code(paths["class_name"], brief, duration, code, feedback)
        issues = manim_lint.lint_code(code, paths["class_name"])
        if not issues:
            break
        feedback = "Static check found problems:\n- " + "\n- ".join(issues)
        print(f"  scene {scene_id} code attempt {attempt}: lint failed, retrying")
    else:
        print(f"  scene {scene_id}: WARNING — lint still failing after {MAX_ATTEMPTS} attempts, saving anyway")

    paths["code"].parent.mkdir(parents=True, exist_ok=True)
    paths["code"].write_text(code)
    print(f"  scene {scene_id}: code -> {paths['code']}")
    return code


def run_render_stage(project_dir: Path, scene_id: int) -> Path:
    """Render already-saved code to a silent clip."""
    paths = scene_paths(project_dir, scene_id)
    if not paths["code"].exists():
        raise SystemExit(f"scene {scene_id}: no saved code at {paths['code']} — run the `code` stage first")
    code = paths["code"].read_text()
    video_path = manim_render.render_scene(code, paths["class_name"], paths["work_dir"])
    print(f"  scene {scene_id}: rendered -> {video_path}")
    return video_path


def run_compose_stage(project_dir: Path, scene_id: int, video_path: Path = None) -> Path:
    paths = scene_paths(project_dir, scene_id)
    if video_path is None:
        matches = sorted(paths["work_dir"].glob(f"media/videos/{paths['class_name']}/*/{paths['class_name']}.mp4"))
        if not matches:
            raise SystemExit(f"scene {scene_id}: no rendered clip found under {paths['work_dir']} — run the `render` stage first")
        video_path = matches[0]
    paths["final"].parent.mkdir(parents=True, exist_ok=True)
    av_compose.compose_scene(video_path, paths["audio"], paths["final"])
    print(f"  scene {scene_id}: composed -> {paths['final']}")
    return paths["final"]


def run_all_stages(project_dir: Path, scene_id: int) -> None:
    """Full code -> lint -> render -> duration-check loop, then compose."""
    paths = scene_paths(project_dir, scene_id)
    brief = paths["brief"].read_text()
    duration = duration_seconds(paths["audio"])

    code, feedback, video_path = None, None, None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        code = scene_coder.generate_code(paths["class_name"], brief, duration, code, feedback)
        issues = manim_lint.lint_code(code, paths["class_name"])
        if issues:
            feedback = "Static check found problems:\n- " + "\n- ".join(issues)
            print(f"  scene {scene_id} attempt {attempt}: lint failed, retrying")
            continue
        try:
            candidate = manim_render.render_scene(code, paths["class_name"], paths["work_dir"])
        except manim_render.RenderError as exc:
            feedback = f"Render failed: {exc}"
            print(f"  scene {scene_id} attempt {attempt}: render failed, retrying")
            continue
        actual = duration_seconds(candidate)
        shortfall = duration - actual
        if shortfall > DURATION_SHORTFALL_TOLERANCE:
            feedback = (
                f"The rendered clip is only {actual:.1f}s long but needs to reach "
                f"{duration:.1f}s to cover the full narration. Cover more of the brief, "
                f"or add self.wait() time, to fill the remaining {shortfall:.1f}s."
            )
            print(f"  scene {scene_id} attempt {attempt}: too short ({actual:.1f}s vs {duration:.1f}s), retrying")
            continue
        video_path = candidate
        break
    else:
        raise SystemExit(f"scene {scene_id}: failed after {MAX_ATTEMPTS} attempts — {feedback}")

    paths["code"].parent.mkdir(parents=True, exist_ok=True)
    paths["code"].write_text(code)
    run_compose_stage(project_dir, scene_id, video_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True, help="Project folder slug under projects/, e.g. python-course")
    parser.add_argument("--scenes", default=None, help="Comma-separated scene ids (default: all scenes with a prompt file)")
    parser.add_argument("--stage", choices=["all", "code", "render", "compose"], default="all")
    args = parser.parse_args()

    project_dir = PROJECTS_DIR / args.project
    if not project_dir.is_dir():
        raise SystemExit(f"No project folder at {project_dir}")

    scene_ids = [int(s) for s in args.scenes.split(",")] if args.scenes else discover_scenes(project_dir)
    if not scene_ids:
        raise SystemExit(f"No scenes found under {project_dir / 'prompts'}")
    print(f"Project: {args.project}  Stage: {args.stage}  Scenes: {scene_ids}")

    for scene_id in scene_ids:
        print(f"Scene {scene_id}:")
        try:
            if args.stage == "all":
                run_all_stages(project_dir, scene_id)
            elif args.stage == "code":
                run_code_stage(project_dir, scene_id)
            elif args.stage == "render":
                run_render_stage(project_dir, scene_id)
            elif args.stage == "compose":
                run_compose_stage(project_dir, scene_id)
        except SystemExit as exc:
            print(f"  scene {scene_id}: FAILED — {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
