"""Small shared ffmpeg/ffprobe helpers used across the render and compose stages."""

import shutil
import subprocess
from pathlib import Path


def check_ffmpeg_available() -> None:
    """Fails fast, with an actionable message, if ffmpeg/ffprobe can't actually run.

    Without this, a broken install surfaces as a raw CalledProcessError deep into
    stage_prompts/stage_render -- after the script and audio stages (several
    minutes of LLM/TTS work) have already run. Checking upfront turns that into
    an instant, readable failure instead. See README's ffmpeg install section --
    on Windows, conda-forge's shared-library ffmpeg build has been seen to install
    an ffprobe.exe that crashes with STATUS_ENTRYPOINT_NOT_FOUND due to a mismatched
    DLL; a static build (e.g. `winget install Gyan.FFmpeg`) doesn't have that failure
    mode since everything it needs is self-contained.
    """
    for exe in ("ffmpeg", "ffprobe"):
        path = shutil.which(exe)
        if path is None:
            raise SystemExit(
                f"'{exe}' was not found on PATH. Install ffmpeg (see README section 3) "
                f"before running the pipeline."
            )
        try:
            subprocess.run([path, "-version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, OSError) as exc:
            raise SystemExit(
                f"'{exe}' was found at {path} but failed to run ({exc}). The install is "
                f"likely broken/corrupted -- on Windows this often means a conda-forge "
                f"ffmpeg build with a mismatched DLL; switching to a static build (e.g. "
                f"`winget install Gyan.FFmpeg`) and making sure it's the one on PATH "
                f"usually fixes it. See README section 3."
            ) from exc


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
