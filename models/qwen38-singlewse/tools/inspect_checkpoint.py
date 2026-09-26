"""Pin and inspect safetensors headers without downloading model payloads."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SIZES = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "F16": 2}


def main():
    hub = json.loads((ROOT / "configs/hub.json").read_text())
    index = json.loads((ROOT / "configs/model.safetensors.index.json").read_text())["weight_map"]
    files = {s["rfilename"]: s for s in hub["siblings"]}
    base = "https://huggingface.co/" + hub["id"] + "/resolve/" + hub["sha"] + "/"

    def bounded_range(name, start, count):
        req = urllib.request.Request(base + name, headers={"Range": f"bytes={start}-{start+count-1}"})
        with urllib.request.urlopen(req, timeout=30) as response:
            if response.status != 206:
                raise ValueError("Server did not honor byte range")
            expected = f"bytes {start}-{start+count-1}/{files[name]['size']}"
            if response.headers.get("Content-Range") != expected:
                raise ValueError("Range identity mismatch")
            data = response.read(count + 1)
            if len(data) != count:
                raise ValueError("Short/oversize range")
            return data

    def inspect(name):
        n = struct.unpack("<Q", bounded_range(name, 0, 8))[0]
        if not 2 <= n <= 1 << 20:
            raise ValueError("Unsafe header size")
        raw = bounded_range(name, 8, n)
        header = json.loads(raw)
        end = 0
        tensors = {}
        for tensor, spec in sorted(((k, v) for k, v in header.items() if k != "__metadata__"),
                                   key=lambda pair: pair[1]["data_offsets"][0]):
            a, b = spec["data_offsets"]
            if a != end or b-a != math.prod(spec["shape"]) * SIZES[spec["dtype"]]:
                raise ValueError("Tensor extent mismatch: " + tensor)
            if index.get(tensor) != name:
                raise ValueError("Index mismatch: " + tensor)
            tensors[tensor] = dict(spec, shard=name, data_start=n+8, bytes=b-a)
            end = b
        if n+8+end != files[name]["size"]:
            raise ValueError("Shard length mismatch")
        return name, dict(header_sha256=hashlib.sha256(raw).hexdigest(),
                          payload_bytes=end, header_bytes=n, tensors=tensors)

    with ThreadPoolExecutor(max_workers=4) as pool:
        shards = dict(pool.map(inspect, sorted(set(index.values()))))
    tensors = {k: v for shard in shards.values() for k, v in shard["tensors"].items()}
    assert set(tensors) == set(index)
    dtype_bytes = Counter()
    role_bytes = Counter()
    for name, spec in tensors.items():
        dtype_bytes[spec["dtype"]] += spec["bytes"]
        role = "vision" if name.startswith("model.visual.") else "mtp" if name.startswith("mtp.") else "text"
        role_bytes[role] += spec["bytes"]
    result = dict(repository=hub["id"], revision=hub["sha"], tensors=tensors,
                  shards={k: {x: y for x, y in v.items() if x != "tensors"} for k, v in shards.items()},
                  dtype_bytes=dict(dtype_bytes), role_bytes=dict(role_bytes),
                  total_payload_bytes=sum(dtype_bytes.values()), tensor_count=len(tensors),
                  qualification="Header/index consistency only; payload hashes are not yet verified")
    (ROOT / "configs/tensors.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("tensors", "shards")}, indent=2))


if __name__ == "__main__":
    main()
