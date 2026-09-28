"""Checksum manifest (data/MANIFEST.json) so any silent change to a dataset is caught."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: str | Path, raw_files: dict[str, str], lfr_graphs: dict[str, dict]) -> dict:
    manifest = {"raw": dict(sorted(raw_files.items())), "lfr": dict(sorted(lfr_graphs.items()))}
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def verify_raw_file(manifest: dict | None, rel_path: str, abs_path: str | Path) -> bool | None:
    """True/False if the file is in the manifest and matches / mismatches; None if not tracked."""
    if manifest is None or rel_path not in manifest.get("raw", {}):
        return None
    return manifest["raw"][rel_path] == sha256_file(abs_path)
