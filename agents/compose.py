"""Compose stage: merges each scene's video with its narration audio,
length-locked to the audio. Deterministic — no LLM call.

manim/matplotlib scenes use their rendered clip. image_only/audio_only
scenes get a placeholder title card (no illustration generator exists
yet) so the pipeline still produces a complete video end to end.
"""

from pathlib import Path

from tqdm.auto import tqdm

from state import Scene, VideoState
from utils.av_compose import compose_scene
from utils.ffmpeg_utils import duration_seconds
from utils.notify import notify
from utils.placeholder import make_placeholder_video
from utils.storage import project_dir


def _compose_scene(scene: Scene, composed_dir) -> Scene:
    duration = duration_seconds(Path(scene.audio_path))
    out_path = composed_dir / f"scene_{scene.id:02d}.mp4"

    if scene.video_path:
        video_path = scene.video_path
    else:
        placeholder_path = composed_dir / f"scene_{scene.id:02d}_placeholder.mp4"
        video_path = make_placeholder_video(scene.subtopic, duration, placeholder_path)

    composed = compose_scene(video_path, scene.audio_path, out_path)
    return scene.model_copy(update={"composed_path": str(composed)})


def compose_node(state: VideoState) -> dict:
    composed_dir = project_dir(state.topic) / "composed"
    composed_dir.mkdir(parents=True, exist_ok=True)
    updated_scenes = [
        _compose_scene(scene, composed_dir) for scene in tqdm(state.scenes, desc="Compose")
    ]
    notify(f"Compose done — {len(updated_scenes)} scene(s)")
    return {"scenes": updated_scenes}
