"""Extract a publisher-verified FP8 tile and freeze independent error criteria."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np


def decode(codes):
    c = codes.astype(np.uint32)
    e, m = (c & 127) >> 3, c & 7
    value = np.where(e == 0, m * 2.0**-9, (1 + m / 8) * np.exp2(e.astype(np.int32)-7))
    value = np.where(c & 128, -value, value).astype(np.float32)
    value[(c & 127) == 127] = np.nan
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    receipt = json.loads((a.model / "LAYER0-COMPLETE.json").read_text())
    shard = a.model / "layers-0.safetensors"
    pin = next(x for x in receipt["files"] if x["file"] == shard.name)
    h = hashlib.sha256()
    with shard.open("rb") as stream:
        for part in iter(lambda: stream.read(4 << 20), b""): h.update(part)
    assert h.hexdigest() == pin["sha256"]
    with shard.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(length))
    name = "model.language_model.layers.0.mlp.gate_proj.weight"
    scale_name = name + "_scale_inv"
    w, s = header[name], header[scale_name]
    assert w["dtype"] == "F8_E4M3" and s["dtype"] == "BF16"
    matrix = np.memmap(shard, mode="r", offset=8+length+w["data_offsets"][0], dtype=np.uint8, shape=tuple(w["shape"]))
    scale_bf16 = np.memmap(shard, mode="r", offset=8+length+s["data_offsets"][0], dtype="<u2", shape=tuple(s["shape"]))
    codes = np.array(matrix[:272, :128], copy=True)
    scales = (np.array(scale_bf16[:3, 0], dtype=np.uint32) << 16).view(np.float32)
    assert np.isfinite(scales).all() and np.all(scales > 0) and np.all((codes & 127) != 127)
    decoded = decode(np.arange(256, dtype=np.uint8))
    # A second implementation, supplied by PyTorch, independently checks all encodings.
    import torch
    oracle = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()
    finite = np.isfinite(oracle)
    assert np.array_equal(decoded[finite].view(np.uint32), oracle[finite].view(np.uint32))
    actual_weights = decode(codes) * scales[np.arange(272) // 128, None]
    vectors = np.stack([np.sin(np.arange(128)*.17).astype(np.float32),
                        ((np.arange(128)%19-9)/16).astype(np.float32), np.zeros(128,np.float32)])
    exact = vectors.astype(np.float64) @ actual_weights.astype(np.float64).T
    sums = np.abs(vectors.astype(np.float64)) @ np.abs(actual_weights.astype(np.float64)).T
    gamma = (2*128*2**-24)/(1-2*128*2**-24)
    bounds = gamma*sums + np.finfo(np.float32).tiny
    packed = np.ascontiguousarray(codes.T).reshape(-1).view("<u2")
    a.output.mkdir(parents=True, exist_ok=False)
    with (a.output/"fixture.npz").open("xb") as stream:
        np.savez(stream, weights=packed, scales=scales, vectors=vectors, exact=exact,
                 bounds=bounds, decoded=decoded, finite=finite)
    meta = dict(repository=receipt["repository"], revision=receipt["revision"], tensor=name,
                scale_tensor=scale_name, rows=[0,272], columns=[0,128], source=pin,
                fixture_sha256=hashlib.sha256((a.output/"fixture.npz").read_bytes()).hexdigest(),
                all_254_finite_encodings_exact_against_torch=True,
                weight_storage_bytes=packed.nbytes+scales.nbytes, full_model=False,
                numerical_policy="FP64 dot oracle; gamma(2*128)*sum(abs(products)); frozen before CSL output",
                scope="Weight E4M3FN decode and FP32 GEMV; activation quantization not yet qualified")
    (a.output/"fixture.json").write_text(json.dumps(meta, indent=2)+"\n")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__": main()
