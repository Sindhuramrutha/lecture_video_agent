"""Persists pipeline state to disk so every stage's output is inspectable."""

import re
from pathlib import Path

from state import VideoState

PROJECTS_DIR = Path(__file__).resolve().parent.parent / "projects"


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "untitled"


def project_dir(topic: str) -> Path:
    path = PROJECTS_DIR / slugify(topic)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_state(state: VideoState, directory: Path) -> Path:
    out_path = directory / "state.json"
    out_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    return out_path
