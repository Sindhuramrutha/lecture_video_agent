"""Loads user-supplied reference documents into plain text for agent context.

No chunking or embedding — references are concatenated directly into the
prompt. Fine for a handful of short documents; revisit if that stops
being true.
"""

from pathlib import Path

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def list_reference_files(directory: Path) -> list[Path]:
    """Every supported file currently sitting in a references folder."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)


def load_references(paths: list[Path | str]) -> list[str]:
    texts = []
    for path_str in paths:
        path = Path(path_str)
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            text = path.read_text()
        texts.append(text)
    return texts
