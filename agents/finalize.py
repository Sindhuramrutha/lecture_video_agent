"""Finalize stage: stitches all composed scenes into the final video and
generates subtitles. Deterministic — no LLM call.

Subtitles are built from the same word-alignment timestamps that drove
animation timing in the Visual branch (see utils/subtitles.py), not a
second ASR pass on the final mix — one source of truth for timing
means audio, video, and subtitles can't drift out of sync with each
other.
"""

from state import VideoState
from utils.notify import notify
from utils.storage import project_dir
from utils.stitch import stitch_clips
from utils.subtitles import build_srt


def finalize_node(state: VideoState) -> dict:
    directory = project_dir(state.topic)

    srt_text = build_srt(state.scenes)
    subtitles_path = directory / "subtitles.srt"
    subtitles_path.write_text(srt_text)

    clip_paths = [scene.composed_path for scene in state.scenes if scene.composed_path]
    final_path = stitch_clips(clip_paths, directory / "final.mp4")

    notify("Final video is ready!")
    return {
        "subtitles_path": str(subtitles_path),
        "final_video_path": str(final_path),
    }
