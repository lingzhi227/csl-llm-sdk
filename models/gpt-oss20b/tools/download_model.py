"""Acquire one pinned public checkpoint with bounded RAM and resumable files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import time
import urllib.request

MODEL = "openai/gpt-oss-20b"
REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
SMALL = {"LICENSE", "USAGE_POLICY", "README.md", "chat_template.jinja",
         "config.json", "generation_config.json", "model.safetensors.index.json",
         "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"}


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(root, item):
    name = item["rfilename"]
    assert Path(name).name == name
    size = item["size"]
    expected = item.get("lfs", {}).get("sha256")
    final = root / name
    partial = root / (name + ".partial")
    if final.exists():
        if final.stat().st_size != size:
            raise ValueError(f"Existing size mismatch: {name}")
    else:
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise ValueError(f"Oversized partial: {name}")
        if offset < size:
            # A fresh query avoids expired cached signed CDN redirects on resume.
            url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}?download=true&run={time.time_ns()}"
            request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"})
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                    raise ValueError("Server did not honor exact resume range")
                if not offset and response.status not in (200, 206):
                    raise ValueError("Unexpected download status")
                with partial.open("ab") as stream:
                    last = time.monotonic()
                    while data := response.read(4 << 20):
                        if offset + len(data) > size:
                            raise ValueError("Response exceeds publisher size")
                        stream.write(data)
                        offset += len(data)
                        if time.monotonic() - last >= 15:
                            stream.flush()
                            atomic(root / "progress.json", {"file": name, "bytes": offset, "total": size, "time": time.time()})
                            last = time.monotonic()
                    stream.flush()
                    os.fsync(stream.fileno())
        if partial.stat().st_size != size:
            raise ValueError(f"Incomplete download: {name}")
        actual = sha256(partial)
        if expected and actual != expected:
            raise ValueError(f"Publisher SHA-256 mismatch: {name}")
        partial.replace(final)
    actual = sha256(final)
    if expected and actual != expected:
        raise ValueError(f"Publisher SHA-256 mismatch: {name}")
    # Small Git objects have Git blob SHA-1 rather than LFS SHA-256.
    if not expected:
        raw = final.read_bytes()
        blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if blob != item["blobId"]:
            raise ValueError(f"Publisher Git object mismatch: {name}")
    return {"file": name, "bytes": size, "sha256": actual, "publisher_verified": True}


def audit(root):
    index = json.loads((root / "model.safetensors.index.json").read_text())
    tensors = {}
    for shard in sorted(set(index["weight_map"].values())):
        path = root / shard
        with path.open("rb") as stream:
            n, = struct.unpack("<Q", stream.read(8))
            if n > 8 << 20:
                raise ValueError("Unbounded safetensors header")
            header = json.loads(stream.read(n))
        offsets = []
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            a, b = spec["data_offsets"]
            count = 1
            for dim in spec["shape"]:
                count *= dim
            if spec["dtype"] not in ("BF16", "U8"):
                raise ValueError(f"Unexpected dtype {name}")
            if b - a != count * (2 if spec["dtype"] == "BF16" else 1):
                raise ValueError(f"Bad tensor extent {name}")
            if index["weight_map"].get(name) != shard or name in tensors:
                raise ValueError("Index/header mismatch")
            offsets.append((a, b))
            tensors[name] = dict(spec, file=shard, payload_offset=8+n)
        cursor = 0
        for a, b in sorted(offsets):
            if a != cursor:
                raise ValueError("Tensor gap or overlap")
            cursor = b
        if 8+n+cursor != path.stat().st_size:
            raise ValueError("Shard size/header mismatch")
    if set(tensors) != set(index["weight_map"]):
        raise ValueError("Incomplete checkpoint")
    payload = sum(t["data_offsets"][1]-t["data_offsets"][0] for t in tensors.values())
    if len(tensors) != 459 or payload != 13_761_264_768:
        raise ValueError("Pinned model identity mismatch")
    atomic(root / "tensor-inventory.json", tensors)
    return {"tensors": len(tensors), "payload_bytes": payload}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    root = args.destination.resolve()
    if not str(root).startswith("/srv/model-storage/"):
        raise ValueError("This deployment requires the Mass1 HDD")
    if not os.path.ismount("/srv/model-storage"):
        raise ValueError("HDD not mounted; no SSD fallback")
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (root / "download.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if shutil.disk_usage(root).free < 48 << 30:
        raise ValueError("Insufficient HDD reserve")
    url = f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(url, timeout=30) as stream:
        metadata = json.load(stream)
    if metadata["sha"] != REVISION:
        raise ValueError("Revision mismatch")
    atomic(root / "publisher-metadata.json", metadata)
    selected = [f for f in metadata["siblings"] if f["rfilename"] in SMALL or ("/" not in f["rfilename"] and f["rfilename"].endswith(".safetensors"))]
    if len(selected) != 13:
        raise ValueError("Expected ten support files and three complete shards")
    receipts = []
    for item in sorted(selected, key=lambda f: (f["rfilename"].endswith(".safetensors"), f["rfilename"])):
        receipt = fetch(root, item)
        receipts.append(receipt)
        atomic(root / "download-receipts.json", receipts)
        print(json.dumps(receipt), flush=True)
    atomic(root / "COMPLETE.json", {"model": MODEL, "revision": REVISION,
           "files": receipts, **audit(root), "completed_at": time.time()})


if __name__ == "__main__":
    main()
