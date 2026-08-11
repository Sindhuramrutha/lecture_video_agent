#!/usr/bin/env python
"""Runs the content-to-video pipeline for one chapter, staged so you can review
silent renders before audio gets muxed in, with no LLM-mediated session
required. Wraps the same agent functions the LangGraph workflow uses
(agents/scene_script.py, agents/visual_render.py, etc.).

Point it at a detailed-content file sitting in projects/<project>/input_files/.
Every chapter gets its own subfolder under the project (so re-running a later
chapter never overwrites an earlier one's files); the final stitched video is
promoted to a shared videos/ folder at the project root. Progress between
separate invocations (e.g. running `render`, reviewing, then `compose` later)
is persisted to <chapter-slug>/state.json.

One-halt workflow (recommended -- everything automatic except one review pause
before audio gets muxed in):
  # 1. script -> audio -> prompts -> render, all in one command
  python run_chapter.py --project python-course --input ch3.docx --stage prepare

  # 2. YOU: watch projects/python-course/<chapter-slug>/rendered_video/scene_*_silent.mp4

  # 3. fix anything that looks wrong (repeat per scene as needed)
  python run_chapter.py --project python-course --input ch3.docx \\
      --scenes 2 --revise "The neuron highlight boxes overlap, fade each one out before the next appears"

  # 4. happy with everything -- mux audio in and stitch the final video
  python run_chapter.py --project python-course --input ch3.docx --stage compose

Other usage:
  # run everything straight through, no review pause at all
  python run_chapter.py --project python-course --input ch3.docx --stage all

  # run one stage at a time, for finer-grained control
  python run_chapter.py --project python-course --input ch3.docx --stage script
  python run_chapter.py --project python-course --input ch3.docx --stage audio
  python run_chapter.py --project python-course --input ch3.docx --stage prompts
  python run_chapter.py --project python-course --input ch3.docx --stage render  [--scenes 1,3]
  python run_chapter.py --project python-course --input ch3.docx --stage compose [--scenes 1,3]

Layout produced (chapter-slug is derived from the input filename):
  projects/<project>/input_files/<input>                -- input (you provide this)
  projects/<project>/<chapter-slug>/state.json           -- scene state, resumed across invocations
  projects/<project>/<chapter-slug>/scene-wise-script/   -- narration + animation description per scene
  projects/<project>/<chapter-slug>/prompts/             -- timed animation briefs
  projects/<project>/<chapter-slug>/audio/               -- narration audio (Kokoro + alignment)
  projects/<project>/<chapter-slug>/manim_code/          -- generated Manim source
  projects/<project>/<chapter-slug>/manim/               -- raw Manim render working dirs
  projects/<project>/<chapter-slug>/rendered_video/      -- scene_NN_silent.mp4 (review this) and
                                                             scene_NN_with_audio.mp4 (after compose)
  projects/<project>/videos/<chapter-slug>.mp4           -- final stitched video
  projects/<project>/videos/<chapter-slug>.srt           -- subtitles
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from tqdm.auto import tqdm

# Quiet down library-level chatter that isn't actionable for someone running this
# pipeline standalone (per-request HTTP logs from the LLM clients, HuggingFace Hub's
# auth-rate-limit notice, faster-whisper's internal alignment warnings) -- it drowns
# out the WARNING/ERROR lines that actually mean something went wrong with a scene.
# Must run before importing anything that configures its own logger (Kokoro/faster-whisper).
for _noisy_logger in ["httpx", "httpcore", "openai", "huggingface_hub", "faster_whisper"]:
    logging.getLogger(_noisy_logger).setLevel(logging.ERROR)

from agents.prompt_maker import make_render_prompt
from agents.scene_script import scene_script_node
from agents.visual_render import RENDERERS, build_chapter_context
from state import Scene, VideoState
from utils.audio_alignment import align
from utils.av_compose import compose_scene
from utils.ffmpeg_utils import check_ffmpeg_available, duration_seconds
from utils.llm import generate_code_model
from utils.placeholder import make_placeholder_video
from utils.references import load_references
from utils.stitch import stitch_clips
from utils.storage import slugify
from utils.subtitles import build_srt
from utils.tts import synthesize

_EXPLAIN_SYSTEM = """You are helping someone with no programming background understand why an \
automated video-rendering step failed, using only the raw error text they give you. Assume they \
cannot read a Python traceback. Respond in two short parts:
1. What went wrong, in one or two plain-English sentences -- no jargon, no code terms unless \
you also say what they mean.
2. One ready-to-use sentence of feedback text they can paste directly after --revise "..." to \
fix it -- concrete and specific to this error, not generic advice.
Nothing else. No preamble, no "I understand you're facing an issue"."""

PROJECTS_DIR = Path(__file__).resolve().parent / "projects"


def load_scenes(chapter_dir: Path) -> Optional[list[Scene]]:
    state_path = chapter_dir / "state.json"
    if not state_path.exists():
        return None
    return [Scene(**s) for s in json.loads(state_path.read_text(encoding="utf-8"))]


def save_scenes(chapter_dir: Path, scenes: list[Scene]) -> list[Scene]:
    state_path = chapter_dir / "state.json"
    state_path.write_text(json.dumps([s.model_dump() for s in scenes], indent=2), encoding="utf-8")
    return scenes


def stage_script(chapter_dir: Path, state: VideoState) -> list[Scene]:
    scenes = scene_script_node(state)["scenes"]
    script_dir = chapter_dir / "scene-wise-script"
    script_dir.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        text = (
            f"Subtopic: {scene.subtopic}\n"
            f"Scene type: {scene.scene_type}\n\n"
            f"Narration:\n{scene.narration}\n\n"
            f"Animation description:\n{scene.animation_description}\n"
        )
        (script_dir / f"scene_{scene.id:02d}.txt").write_text(text, encoding="utf-8")
    print(f"  scene script: {len(scenes)} scene(s) -> {script_dir}")
    return save_scenes(chapter_dir, scenes)


def stage_audio(chapter_dir: Path, scenes: list[Scene]) -> list[Scene]:
    audio_dir = chapter_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir = chapter_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    updated = []
    for scene in tqdm(scenes, desc="Audio", unit="scene"):
        out_path = audio_dir / f"scene_{scene.id:02d}.wav"
        synthesize(scene.narration, out_path)
        timestamps = align(out_path)
        # The word-level alignment isn't just a byproduct -- prompt_maker uses these
        # exact timestamps to tie each animation beat to what's being said at that
        # moment, so it's worth keeping in a form you can actually inspect.
        transcript_lines = [f"{w.start:.2f}\t{w.end:.2f}\t{w.word}" for w in timestamps]
        (transcripts_dir / f"scene_{scene.id:02d}.txt").write_text("\n".join(transcript_lines), encoding="utf-8")
        updated.append(scene.model_copy(update={"audio_path": str(out_path), "timestamps": timestamps}))
    print(f"  audio: {len(updated)} scene(s) -> {audio_dir}")
    print(f"  transcripts -> {transcripts_dir}")
    return save_scenes(chapter_dir, updated)


def stage_prompts(chapter_dir: Path, scenes: list[Scene]) -> list[Scene]:
    prompts_dir = chapter_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    updated = []
    previous_scene = None
    for scene in tqdm(scenes, desc="Prompts", unit="scene"):
        render_prompt = make_render_prompt(scene, previous_scene) if scene.scene_type in RENDERERS else None
        if render_prompt:
            (prompts_dir / f"scene_{scene.id:02d}_prompt.txt").write_text(render_prompt, encoding="utf-8")
        scene = scene.model_copy(update={"render_prompt": render_prompt})
        updated.append(scene)
        previous_scene = scene
    print(f"  prompts -> {prompts_dir}")
    return save_scenes(chapter_dir, updated)


def _log_scene_error(chapter_dir: Path, scene_id: int, detail: str) -> Path:
    """Saves the full failure detail to a file. The terminal only ever shows a
    one-line gist (see _short_gist) -- a full Manim traceback is many lines and
    floods a live run; the file is there when you actually need to dig in, or
    want to paste it somewhere for help."""
    logs_dir = chapter_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"scene_{scene_id:02d}_error.log"
    log_path.write_text(detail, encoding="utf-8")
    return log_path


def stage_explain(chapter_dir: Path, scene_id: int) -> None:
    """Asks the code-generation LLM (DeepSeek, same one already configured for
    rendering -- no separate account or key needed) to translate a scene's saved
    error log into plain English plus a ready-to-use --revise suggestion. For
    failures that aren't one of the specific known patterns already explained
    automatically (a duration mismatch, a lint issue), this is the answer to
    "what do I do when Claude isn't around to read the traceback for me."
    """
    log_path = chapter_dir / "logs" / f"scene_{scene_id:02d}_error.log"
    if not log_path.exists():
        raise SystemExit(
            f"No saved error log for scene {scene_id} at {log_path} -- "
            f"that scene hasn't failed (or hasn't been run yet)."
        )
    detail = log_path.read_text(encoding="utf-8")
    print(f"Asking DeepSeek to explain scene {scene_id}'s failure...\n")
    explanation = generate_code_model(system=_EXPLAIN_SYSTEM, prompt=detail, max_tokens=400)
    print(explanation)


def _short_gist(detail: str, limit: int = 180) -> str:
    """The single most useful line of a failure detail for a live-terminal summary:
    the last non-blank line, which for a Python traceback is the exception itself
    (e.g. "TypeError: ...") and for our own duration/lint messages is the message.
    """
    lines = [l.strip(" │╭╮╰╯─┃┏┓┗┛\t") for l in detail.strip().split("\n") if l.strip()]
    gist = lines[-1] if lines else detail.strip()
    return gist[:limit] + ("…" if len(gist) > limit else "")


def _print_stage_summary(
    stage: str, scenes: list[Scene], scene_ids: Optional[list[int]], project: str, input_file: str, is_success
) -> None:
    """is_success: Scene -> bool, what "this scene is done" means for this stage
    (rendered successfully for `render`, muxed with audio for `compose`)."""
    targets = scenes if scene_ids is None else [s for s in scenes if s.id in scene_ids]
    attempted = [s for s in targets if s.scene_type in RENDERERS]
    failed = [s for s in attempted if not is_success(s)]
    succeeded = len(attempted) - len(failed)
    print(f"\n{'=' * 60}\n{stage.upper()} SUMMARY: {succeeded}/{len(attempted)} scene(s) OK", end="")
    if not failed:
        print(" — all good.")
    else:
        print()
        for s in failed:
            print(f"  scene {s.id}: {_short_gist(s.last_feedback or '(no detail)')}")
            print(f"    full detail: {chapter_dir_for(project, input_file) / 'logs' / f'scene_{s.id:02d}_error.log'}")
            print(
                f"    fix:  python run_chapter.py --project {project} --input {input_file} "
                f"--scenes {s.id} --revise \"describe what's wrong\""
            )
    print(f"{'=' * 60}\n")


def chapter_dir_for(project: str, input_file: str) -> Path:
    return PROJECTS_DIR / project / slugify(Path(input_file).stem)


def stage_render(chapter_dir: Path, scenes: list[Scene], scene_ids: Optional[list[int]] = None) -> list[Scene]:
    """Renders silent clips only -- no audio. Review rendered_video/scene_NN_silent.mp4
    before running the `compose` stage.
    """
    rendered_dir = chapter_dir / "rendered_video"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    chapter_context = build_chapter_context(scenes)
    updated = []
    previous_scene = None
    bar = tqdm(scenes, desc="Render", unit="scene")
    for scene in bar:
        if scene_ids is not None and scene.id not in scene_ids:
            updated.append(scene)
            previous_scene = scene
            continue
        bar.set_postfix_str(f"scene {scene.id} ({scene.scene_type})")
        if scene.scene_type in RENDERERS:
            try:
                scene = RENDERERS[scene.scene_type](
                    scene, chapter_dir, previous_scene=previous_scene, chapter_context=chapter_context
                )
            except Exception as exc:
                # _run_render_loop only retries manim_render.RenderError (an actual Manim
                # crash) -- anything else (a DeepSeek API hiccup, a network blip, a bug)
                # would otherwise propagate out of this loop and kill every scene still
                # queued in the batch. Catching broadly here means one scene's unexpected
                # failure never takes down scenes that would've rendered fine.
                scene = scene.model_copy(update={
                    "render_status": "failed",
                    "last_feedback": f"Unexpected error (not a Manim render failure): {exc}",
                })
                tqdm.write(f"  scene {scene.id}: ERROR — {exc} — skipping, other scenes continue")
            if scene.render_status == "rendered":
                silent_copy = rendered_dir / f"scene_{scene.id:02d}_silent.mp4"
                shutil.copyfile(scene.video_path, silent_copy)
                tqdm.write(f"  scene {scene.id}: rendered (silent, no audio yet) -> {silent_copy}")
            else:
                detail = scene.last_feedback or "(no detail)"
                log_path = _log_scene_error(chapter_dir, scene.id, detail)
                tqdm.write(f"  scene {scene.id}: FAILED — {_short_gist(detail)}  (full detail: {log_path})")
        else:
            tqdm.write(f"  scene {scene.id}: scene_type={scene.scene_type}, no visual render step; a title card is used at compose time")
        updated.append(scene)
        previous_scene = scene
    return save_scenes(chapter_dir, updated)


def stage_revise(chapter_dir: Path, scenes: list[Scene], scene_id: int, feedback: str) -> list[Scene]:
    """Regenerates one scene's code from your plain-English feedback plus its
    existing code, then re-renders it -- the same manual revise loop
    agents/visual_render.py supports via its `instruction` argument.
    """
    rendered_dir = chapter_dir / "rendered_video"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    chapter_context = build_chapter_context(scenes)
    updated = []
    found = False
    previous_scene = None
    for scene in scenes:
        if scene.id != scene_id:
            updated.append(scene)
            previous_scene = scene
            continue
        found = True
        if scene.scene_type not in RENDERERS:
            raise SystemExit(f"scene {scene_id} is scene_type={scene.scene_type}, nothing to revise")
        print(f"  scene {scene.id}: revising with feedback: {feedback!r}")
        try:
            scene = RENDERERS[scene.scene_type](
                scene, chapter_dir, instruction=feedback, previous_scene=previous_scene, chapter_context=chapter_context
            )
        except Exception as exc:
            scene = scene.model_copy(update={
                "render_status": "failed",
                "last_feedback": f"Unexpected error (not a Manim render failure): {exc}",
            })
            print(f"    ERROR — {exc}")
        if scene.render_status == "rendered":
            silent_copy = rendered_dir / f"scene_{scene.id:02d}_silent.mp4"
            shutil.copyfile(scene.video_path, silent_copy)
            print(f"    revised render -> {silent_copy}")
            print(f"    once it looks right: --stage compose --scenes {scene_id}")
        else:
            detail = scene.last_feedback or "(no detail)"
            log_path = _log_scene_error(chapter_dir, scene.id, detail)
            print(f"    FAILED — {_short_gist(detail)}  (full detail: {log_path})")
        updated.append(scene)
    if not found:
        raise SystemExit(f"No scene with id {scene_id} in this chapter's state")
    return save_scenes(chapter_dir, updated)


def stage_compose(chapter_dir: Path, scenes: list[Scene], scene_ids: Optional[list[int]] = None) -> list[Scene]:
    """Muxes each scene's (already-reviewed) silent clip with its narration audio."""
    rendered_dir = chapter_dir / "rendered_video"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    updated = []
    for scene in tqdm(scenes, desc="Compose", unit="scene"):
        if scene_ids is not None and scene.id not in scene_ids:
            updated.append(scene)
            continue
        padded = f"{scene.id:02d}"
        duration = duration_seconds(Path(scene.audio_path))

        if scene.scene_type in RENDERERS:
            if not scene.video_path:
                tqdm.write(f"  scene {scene.id}: no rendered clip yet — run --stage render first")
                updated.append(scene)
                continue
            video_path = Path(scene.video_path)
        else:
            # image_only / audio_only: no illustration generator exists yet -- a
            # title card keeps the pipeline producing a complete video end to end
            # (same fallback agents/compose.py uses in the LangGraph path).
            video_path = make_placeholder_video(
                scene.subtopic, duration, rendered_dir / f"scene_{padded}_placeholder.mp4"
            )

        out_path = rendered_dir / f"scene_{padded}_with_audio.mp4"
        try:
            compose_scene(video_path, Path(scene.audio_path), out_path)
        except Exception as exc:
            # compose_scene shells out to ffmpeg (subprocess.run(..., check=True)) --
            # a corrupted clip or a transient ffmpeg failure would otherwise crash the
            # whole compose stage instead of just leaving this one scene un-composed.
            log_path = _log_scene_error(chapter_dir, scene.id, str(exc))
            tqdm.write(f"  scene {scene.id}: ERROR composing — {_short_gist(str(exc))}  (full detail: {log_path})")
            updated.append(scene)
            continue
        tqdm.write(f"  scene {scene.id}: composed -> {out_path}")
        updated.append(scene.model_copy(update={"composed_path": str(out_path)}))
    return save_scenes(chapter_dir, updated)


def finalize(project_dir: Path, chapter_slug: str, scenes: list[Scene]) -> None:
    videos_dir = project_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    clip_paths = [s.composed_path for s in scenes if s.composed_path]
    if not clip_paths:
        raise SystemExit("No scenes composed successfully — nothing to stitch.")
    if len(clip_paths) != len(scenes):
        print(f"  NOTE: only {len(clip_paths)}/{len(scenes)} scenes are composed — stitching what's ready.")
    final_path = stitch_clips(clip_paths, videos_dir / f"{chapter_slug}.mp4")

    srt_text = build_srt(scenes)
    (videos_dir / f"{chapter_slug}.srt").write_text(srt_text, encoding="utf-8")

    print(f"Final video -> {final_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True, help="Project folder slug under projects/, e.g. python-course")
    parser.add_argument("--input", required=True, help="Filename under projects/<project>/input_files/")
    parser.add_argument("--topic", default=None, help="Defaults to the input filename")
    parser.add_argument("--audience", default="students new to the topic")
    parser.add_argument("--duration-minutes", type=int, default=15)
    parser.add_argument(
        "--stage", choices=["all", "prepare", "script", "audio", "prompts", "render", "compose"], default="all",
        help=(
            "'prepare' runs script+audio+prompts+render in one command and stops there for "
            "review (the one-halt workflow); 'compose' is the other side of that halt. 'all' "
            "runs everything through to the final video with no pause."
        ),
    )
    parser.add_argument("--scenes", default=None, help="Comma-separated scene ids (render/compose only; default: all)")
    parser.add_argument(
        "--revise", default=None, metavar="FEEDBACK",
        help="Revise one scene's code with this feedback text (requires --scenes <single id>)",
    )
    parser.add_argument(
        "--explain", action="store_true",
        help="Ask DeepSeek to explain a failed scene's saved error log in plain English, plus "
        "suggest --revise feedback (requires --scenes <single id>). For failures that aren't "
        "already explained automatically -- no separate account needed, reuses the same key "
        "already configured for rendering.",
    )
    parser.add_argument(
        "--set-scene-type", default=None, choices=["manim", "matplotlib", "image_only", "audio_only"],
        help="Override one scene's type (requires --scenes <single id>) -- e.g. the classifier "
        "picked image_only for a scene with substantial narration, and you want real animation "
        "instead. Clears that scene's stale render state (code/video/status) so it starts fresh "
        "under the new type; follow up with --stage render --scenes <id> to actually render it.",
    )
    args = parser.parse_args()

    check_ffmpeg_available()

    project_dir = PROJECTS_DIR / args.project
    input_path = project_dir / "input_files" / args.input
    if not input_path.exists():
        raise SystemExit(f"No input file at {input_path}")

    chapter_slug = slugify(input_path.stem)
    chapter_dir = project_dir / chapter_slug
    chapter_dir.mkdir(parents=True, exist_ok=True)

    if args.explain:
        scene_ids = [int(s) for s in args.scenes.split(",")] if args.scenes else None
        if not scene_ids or len(scene_ids) != 1:
            raise SystemExit("--explain requires exactly one scene id via --scenes <id>")
        stage_explain(chapter_dir, scene_ids[0])
        return
    print(f"Chapter: {chapter_slug}  (input: {input_path.name})")

    scene_ids = [int(s) for s in args.scenes.split(",")] if args.scenes else None

    if args.set_scene_type is not None:
        if not scene_ids or len(scene_ids) != 1:
            raise SystemExit("--set-scene-type requires exactly one scene id via --scenes <id>")
        scenes = load_scenes(chapter_dir)
        if scenes is None:
            raise SystemExit(f"No saved state for chapter {chapter_slug} yet — run the pipeline first")
        scene_id = scene_ids[0]
        updated, found = [], False
        for scene in scenes:
            if scene.id == scene_id:
                found = True
                old_type = scene.scene_type
                scene = scene.model_copy(update={
                    "scene_type": args.set_scene_type,
                    "render_prompt": None,
                    "code": None,
                    "video_path": None,
                    "render_attempts": 0,
                    "render_status": "not_applicable",
                    "last_feedback": None,
                    "composed_path": None,
                })
                print(f"  scene {scene_id}: scene_type {old_type!r} -> {args.set_scene_type!r}, render state cleared")
            updated.append(scene)
        if not found:
            raise SystemExit(f"No scene with id {scene_id} in this chapter's state")
        save_scenes(chapter_dir, updated)
        print(f"  next: python run_chapter.py --project {args.project} --input {args.input} --scenes {scene_id} --stage render")
        return

    if args.revise is not None:
        if not scene_ids or len(scene_ids) != 1:
            raise SystemExit("--revise requires exactly one scene id via --scenes <id>")
        scenes = load_scenes(chapter_dir)
        if scenes is None:
            raise SystemExit(f"No saved state for chapter {chapter_slug} yet — run the pipeline first")
        stage_revise(chapter_dir, scenes, scene_ids[0], args.revise)
        return

    def build_state() -> VideoState:
        content = load_references([input_path])[0]
        topic = args.topic or input_path.stem
        return VideoState(
            topic=topic, audience=args.audience, duration_minutes=args.duration_minutes, detailed_script=content
        )

    def render_ok(s: Scene) -> bool:
        return s.render_status == "rendered"

    def compose_ok(s: Scene) -> bool:
        return s.composed_path is not None

    if args.stage == "all":
        scenes = stage_script(chapter_dir, build_state())
        scenes = stage_audio(chapter_dir, scenes)
        scenes = stage_prompts(chapter_dir, scenes)
        scenes = stage_render(chapter_dir, scenes)
        _print_stage_summary("render", scenes, None, args.project, args.input, render_ok)
        scenes = stage_compose(chapter_dir, scenes)
        _print_stage_summary("compose", scenes, None, args.project, args.input, compose_ok)
        finalize(project_dir, chapter_slug, scenes)
        return

    if args.stage == "prepare":
        scenes = stage_script(chapter_dir, build_state())
        scenes = stage_audio(chapter_dir, scenes)
        scenes = stage_prompts(chapter_dir, scenes)
        scenes = stage_render(chapter_dir, scenes)
        _print_stage_summary("render", scenes, None, args.project, args.input, render_ok)
        print(
            f"Ready for review: watch {chapter_dir / 'rendered_video'}/scene_*_silent.mp4\n"
            f"  fix a scene:  python run_chapter.py --project {args.project} --input {args.input} "
            f"--scenes N --revise \"...\"\n"
            f"  when done:    python run_chapter.py --project {args.project} --input {args.input} --stage compose"
        )
        return

    if args.stage == "script":
        stage_script(chapter_dir, build_state())
        return

    scenes = load_scenes(chapter_dir)
    if scenes is None:
        raise SystemExit(f"No saved state for chapter {chapter_slug} yet — run --stage script first")

    if args.stage == "audio":
        stage_audio(chapter_dir, scenes)
    elif args.stage == "prompts":
        stage_prompts(chapter_dir, scenes)
    elif args.stage == "render":
        scenes = stage_render(chapter_dir, scenes, scene_ids)
        _print_stage_summary("render", scenes, scene_ids, args.project, args.input, render_ok)
    elif args.stage == "compose":
        scenes = stage_compose(chapter_dir, scenes, scene_ids)
        _print_stage_summary("compose", scenes, scene_ids, args.project, args.input, compose_ok)
        if scene_ids is None:
            finalize(project_dir, chapter_slug, scenes)
        else:
            print("  (composed a subset — run --stage compose with no --scenes once every scene is ready, to stitch the final video)")


if __name__ == "__main__":
    main()
