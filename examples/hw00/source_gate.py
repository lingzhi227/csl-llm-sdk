"""The admission hash binds all source and prepared bytes before allocation."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent


def verify():
    raw = (ROOT / "manifest.json").read_bytes()
    admission = json.loads((ROOT / "admission.json").read_text())
    if admission["manifest_sha256"] != hashlib.sha256(raw).hexdigest() or admission["candidate"] != "hw00-b-001":
        raise ValueError("Controller admission does not bind this candidate")
    for name, entry in json.loads(raw)["files"].items():
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError("Frozen file extent mismatch: " + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Frozen file hash mismatch: " + name)
    return admission
