"""Small shared ffmpeg/ffprobe helpers used across the render and compose stages."""

import subprocess
from pathlib import Path


def duration_seconds(media_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(media_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())
