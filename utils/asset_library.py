"""A manifest-based library of reusable assets (audio, video, images).

Check here before generating something new — e.g. a standard intro sting
or a recurring background. Reused assets cost no render time and no LLM
tokens, and keep a consistent look across videos.

Nothing generates real assets yet, so the manifest starts empty. Drop a
file under the matching assets/<type>/ folder and call register_asset()
to add it.
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
    raw = json.loads(MANIFEST_PATH.read_text())
    return {key: AssetEntry(**value) for key, value in raw.items()}


def _save_manifest(manifest: dict[str, AssetEntry]) -> None:
    MANIFEST_PATH.write_text(
        json.dumps({key: entry.model_dump() for key, entry in manifest.items()}, indent=2)
    )


def get_asset(key: str) -> Optional[Path]:
    """Returns the asset's path if it's in the library and the file exists, else None."""
    entry = _load_manifest().get(key)
    if entry is None:
        return None
    path = LIBRARY_DIR / entry.path
    return path if path.exists() else None


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
