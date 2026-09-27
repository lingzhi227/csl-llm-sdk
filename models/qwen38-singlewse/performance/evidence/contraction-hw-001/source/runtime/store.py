"""Durable immutable checkpoints with streamed hashing and an atomic cursor."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import uuid


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    with temp.open("xb") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    sync_directory(path.parent)


class Store:
    def __init__(self, root):
        self.root = Path(root).absolute()
        if any(p.is_symlink() for p in (self.root, *self.root.parents)):
            raise ValueError("Checkpoint root must not traverse symlinks")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "snapshots").mkdir(exist_ok=True)
        (self.root / "attempts").mkdir(exist_ok=True)

    @contextmanager
    def locked(self):
        with (self.root / "writer.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def cursor(self):
        return json.loads((self.root / "CURSOR.json").read_text())

    def update(self, cursor):
        atomic_json(self.root / "CURSOR.json", cursor)

    def commit(self, attempt, identity, schema, paths, receipt):
        if not re.fullmatch(r"[0-9a-f]{32}", attempt) or set(paths) != set(schema):
            raise ValueError("Exact attempt and complete tensor coverage required")
        pending = self.root / "snapshots" / (".pending-" + attempt)
        pending.mkdir()  # Never silently reuse an incomplete attempt.
        tensors = {}
        for name, spec in sorted(schema.items()):
            source = Path(paths[name])
            if source.is_symlink() or not source.is_file() or source.stat().st_size != spec.nbytes:
                raise ValueError(f"Invalid tensor extent: {name}")
            out = pending / (name + ".bin")
            h = hashlib.sha256()
            count = 0
            with source.open("rb") as reader, out.open("xb") as writer:
                for chunk in iter(lambda: reader.read(1 << 20), b""):
                    count += len(chunk)
                    if count > spec.nbytes:
                        raise ValueError("Tensor grew during checkpoint")
                    writer.write(chunk)
                    h.update(chunk)
                writer.flush()
                os.fsync(writer.fileno())
            if count != spec.nbytes:
                raise ValueError("Tensor shrank during checkpoint")
            tensors[name] = dict(spec.metadata(), file=out.name, sha256=h.hexdigest())
        manifest = dict(identity, tensors=tensors, receipt=receipt, attempt=attempt)
        atomic_json(pending / "manifest.json", manifest)
        key = hashlib.sha256(encoded(manifest)).hexdigest()
        os.rename(pending, self.root / "snapshots" / key)
        sync_directory(self.root / "snapshots")
        return key

    def load(self, key, identity, schema):
        if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("Invalid checkpoint key")
        folder = self.root / "snapshots" / key
        raw = (folder / "manifest.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != key:
            raise ValueError("Manifest hash mismatch")
        value = json.loads(raw)
        if any(value.get(k) != v for k, v in identity.items()) or set(value["tensors"]) != set(schema):
            raise ValueError("Checkpoint identity or tensor set mismatch")
        paths = {}
        for name, spec in schema.items():
            meta = value["tensors"][name]
            if any(meta.get(k) != v for k, v in spec.metadata().items()) or meta["file"] != name + ".bin":
                raise ValueError("Tensor shape, dtype or storage mismatch")
            path = folder / meta["file"]
            if path.is_symlink() or path.stat().st_size != spec.nbytes or digest(path) != meta["sha256"]:
                raise ValueError("Tensor hash or byte count mismatch")
            paths[name] = path
        return value, paths
