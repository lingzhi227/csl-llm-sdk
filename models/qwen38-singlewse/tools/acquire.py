"""Acquire pinned public files on HDD with publisher verification and a lock."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hub", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--layer0-only", action="store_true")
    p.add_argument("--storage-tier",choices=['mass1','alcf'],default='mass1')
    p.add_argument("--max-mib-per-second",type=float,default=0)
    a = p.parse_args()
    destination=str(a.root.resolve())
    if a.storage_tier=='mass1':
        if not destination.startswith('/srv/model-storage/'):raise ValueError('Model storage must be on Mass1')
    elif destination!='/srv/qwen38-singlewse-hardware/model':
        raise ValueError('Unexpected task-owned ALCF model path')
    if not 0<=a.max_mib_per_second<=64:raise ValueError('Download bandwidth limit')
    a.root.mkdir(parents=True, exist_ok=True)
    lock = (a.root / "download.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    hub = json.loads(a.hub.read_text())
    small = {"config.json", "model.safetensors.index.json", "generation_config.json",
             "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "LICENSE", "README.md"}
    items = [s for s in hub["siblings"] if s["rfilename"] in small or
             (s["rfilename"] == "layers-0.safetensors" if a.layer0_only else s["rfilename"].endswith(".safetensors"))]
    if shutil.disk_usage(a.root).free < sum(x["size"] for x in items) + (32 << 30):
        raise ValueError("Insufficient HDD reserve")
    receipts = []
    for item in items:
        name = item["rfilename"]
        assert Path(name).name == name
        path = a.root / name
        size = item["size"]
        expected = item.get("lfs", {}).get("sha256")
        part = a.root / (name+".partial")
        if not path.exists():
            offset = part.stat().st_size if part.exists() else 0
            assert offset <= size
            if offset < size:
                url = f"https://huggingface.co/{hub['id']}/resolve/{hub['sha']}/{name}?download=true&run={time.time_ns()}"
                request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"})
                initial=offset;started=time.monotonic();last_progress=0
                with urllib.request.urlopen(request, timeout=45) as response:
                    if offset and (response.status != 206 or response.headers.get("Content-Range") != f"bytes {offset}-{size-1}/{size}"):
                        raise ValueError("Unsafe resume range")
                    with part.open("ab") as out:
                        while block := response.read(4 << 20):
                            offset += len(block)
                            if offset > size:
                                raise ValueError("Oversize response")
                            out.write(block)
                            if a.max_mib_per_second:
                                delay=(offset-initial)/(a.max_mib_per_second*(1<<20))-(time.monotonic()-started)
                                if delay>0:time.sleep(delay)
                            if time.monotonic()-last_progress>5:
                                progress=dict(file=name,downloaded_bytes=offset,total_bytes=size,completed_files=len(receipts),bandwidth_limit_mib_s=a.max_mib_per_second)
                                (a.root/'PROGRESS.json').write_text(json.dumps(progress)+'\n');last_progress=time.monotonic()
                        out.flush()
                        os.fsync(out.fileno())
            candidate = part
        else:
            candidate = path
        assert candidate.stat().st_size == size
        sha = digest(candidate)
        if expected:
            assert sha == expected, name
        else:
            raw = candidate.read_bytes()
            assert hashlib.sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest() == item["blobId"], name
        if candidate == part:
            part.replace(path)
        receipts.append(dict(file=name, bytes=size, sha256=sha, publisher_verified=True))
        (a.root / "download-receipts.json").write_text(json.dumps(receipts, indent=2)+"\n")
        print(json.dumps(receipts[-1]), flush=True)
    (a.root / ("LAYER0-COMPLETE.json" if a.layer0_only else "COMPLETE.json")).write_text(
        json.dumps(dict(repository=hub["id"], revision=hub["sha"], files=receipts), indent=2)+"\n")


if __name__ == "__main__":
    main()
