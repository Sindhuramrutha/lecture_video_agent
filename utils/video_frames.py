"""Samples still frames from a rendered clip for the critique step."""

import subprocess
from pathlib import Path

from utils.ffmpeg_utils import duration_seconds


def extract_frames(video_path: Path, n: int = 3) -> list[Path]:
    duration = duration_seconds(video_path)
    out_dir = video_path.parent / f"{video_path.stem}_frames"
    out_dir.mkdir(exist_ok=True)

    frame_paths = []
    for i in range(n):
        timestamp = duration * (i + 0.5) / n
        out_path = out_dir / f"frame_{i}.png"
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(timestamp),
                "-i", str(video_path),
                "-frames:v", "1",
                str(out_path),
            ],
            check=True,
            capture_output=True,
        )
        frame_paths.append(out_path)
    return frame_paths
