"""Verify the frozen sources for an individual experiment."""
import hashlib
import json
from pathlib import Path


def verify():
    root = Path.cwd()
    manifest = json.loads((root / "source-manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen source changed: " + name)
    return manifest
