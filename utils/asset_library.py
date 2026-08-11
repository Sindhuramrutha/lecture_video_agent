"""A manifest-based library of reusable assets (audio, video, images).

Check here before generating something new — e.g. a standard intro sting
or a recurring background. Reused assets cost no render time and no LLM
tokens, and keep a consistent look across videos.

Image assets (see list_image_assets()/image_asset_paths()) are what
agents/scene_coder.py offers the model as an alternative to hand-drawing
recognizable objects (a heart, a brain) from Manim primitives -- see
manim_lint.py's validation of ImageMobject/SVGMobject usage against this
same manifest. Drop a file under assets/<type>/ and call register_asset()
to add one.
"""

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

LIBRARY_DIR = Path(__file__).resolve().parent.parent / "assets"
MANIFEST_PATH = LIBRARY_DIR / "library.json"


class AssetEntry(BaseModel):
    path: str
    asset_type: str  # "audio" | "video" | "image"
    description: str


def _load_manifest() -> dict[str, AssetEntry]:
    if not MANIFEST_PATH.exists():
        return {}
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {key: AssetEntry(**value) for key, value in raw.items()}


def _save_manifest(manifest: dict[str, AssetEntry]) -> None:
    MANIFEST_PATH.write_text(
        json.dumps({key: entry.model_dump() for key, entry in manifest.items()}, indent=2),
        encoding="utf-8",
    )


def get_asset(key: str) -> Optional[Path]:
    """Returns the asset's path if it's in the library and the file exists, else None."""
    entry = _load_manifest().get(key)
    if entry is None:
        return None
    path = LIBRARY_DIR / entry.path
    return path if path.exists() else None


def list_image_assets() -> dict[str, AssetEntry]:
    """Only the image/SVG assets -- the ones scene_coder can reference in generated code."""
    return {key: entry for key, entry in _load_manifest().items() if entry.asset_type == "image"}


def image_asset_paths() -> dict[str, str]:
    """key -> absolute file path, for every registered image asset that exists on disk.

    Used both to inject an ASSET_PATHS dict into generated code and, by
    manim_lint.py, to validate that a scene only references keys that are
    actually real -- never a hallucinated filename.
    """
    return {key: str(path) for key, entry in list_image_assets().items() if (path := get_asset(key)) is not None}


def register_asset(key: str, file_path: Path, asset_type: str, description: str) -> Path:
    file_path = Path(file_path)
    dest_dir = LIBRARY_DIR / asset_type
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file_path.name
    dest.write_bytes(file_path.read_bytes())

    manifest = _load_manifest()
    manifest[key] = AssetEntry(
        path=f"{asset_type}/{file_path.name}", asset_type=asset_type, description=description
    )
    _save_manifest(manifest)
    return dest
