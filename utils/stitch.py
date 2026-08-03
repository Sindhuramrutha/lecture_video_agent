"""Concatenates composed per-scene clips into the final video."""

import subprocess
from pathlib import Path


def stitch_clips(clip_paths: list[Path], out_path: Path) -> Path:
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    list_path = out_path.parent / "concat_list.txt"
    list_path.write_text("\n".join(f"file '{Path(p).resolve()}'" for p in clip_paths))

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )
    return out_path
