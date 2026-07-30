"""Track which files have already been ingested, so we never duplicate them."""

import json

from utils.config import INDEX_DIR

MANIFEST_FILE = INDEX_DIR / "manifest.json"


def load_manifest() -> list[str]:
    if MANIFEST_FILE.exists():
        return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    return []


def mark_ingested(filename: str):
    files = load_manifest()
    if filename not in files:
        files.append(filename)
        MANIFEST_FILE.write_text(json.dumps(files), encoding="utf-8")


def already_ingested(filename: str) -> bool:
    return filename in load_manifest()