"""Merges a scene's silent video with its narration audio, length-locked
to the audio (the source of truth for timing — see the architecture
note on subtitles/sync in agents/finalize.py).

Every composed clip is normalized to the same resolution/fps here —
Manim, Matplotlib, and the Pillow placeholder each produce different
native sizes and frame rates, and concatenating mismatched streams in
Finalize's concat demuxer produces exactly the kind of drift/corruption
(black stretches, audio running past video) this normalization fixes.

If the video is shorter than the audio, the last frame is frozen to
fill the exact gap (not a fixed guess — undershooting the pad leaves
video with no frames while audio keeps playing); if longer, it's
trimmed. Either way the output duration exactly equals the audio
duration.
"""

import subprocess
from pathlib import Path

from utils.ffmpeg_utils import duration_seconds

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_FPS = 24
PAD_BUFFER_SECONDS = 1.0  # safety margin beyond the exact measured gap


def compose_scene(video_path: Path, audio_path: Path, out_path: Path) -> Path:
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio_duration = duration_seconds(audio_path)
    video_duration = duration_seconds(video_path)
    pad = max(0.5, (audio_duration - video_duration) + PAD_BUFFER_SECONDS)

    scale_pad = (
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={TARGET_FPS}"
    )

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-filter_complex", f"[0:v]{scale_pad},tpad=stop_mode=clone:stop_duration={pad:.2f}[v]",
            "-map", "[v]", "-map", "1:a",
            "-t", f"{audio_duration:.3f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )
    return out_path
