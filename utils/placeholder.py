"""Placeholder visual for image_only/audio_only scenes.

Nothing generates real illustrations yet (that's the FLUX/SVG stage,
not built), so this is a plain title-card stopgap so the pipeline can
still produce a complete, watchable video end to end. Replace once
illustration generation exists.

Renders text with Pillow rather than ffmpeg's drawtext filter — this
machine's ffmpeg build doesn't have libfreetype/drawtext compiled in.
"""

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1280, 720
BACKGROUND = (20, 20, 40)
TEXT_COLOR = (230, 230, 230)


def _title_card(text: str, out_path: Path) -> Path:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=48)

    lines = []
    words = text.split()
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if draw.textlength(candidate, font=font) > WIDTH - 160:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)

    line_height = 60
    total_height = line_height * len(lines)
    y = (HEIGHT - total_height) // 2
    for line in lines:
        w = draw.textlength(line, font=font)
        draw.text(((WIDTH - w) / 2, y), line, fill=TEXT_COLOR, font=font)
        y += line_height

    image.save(out_path)
    return out_path


def make_placeholder_video(text: str, duration: float, out_path: Path) -> Path:
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image_path = out_path.parent / f"{out_path.stem}_title.png"
    _title_card(text, image_path)

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(image_path),
            "-t", f"{duration:.2f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )
    return out_path
