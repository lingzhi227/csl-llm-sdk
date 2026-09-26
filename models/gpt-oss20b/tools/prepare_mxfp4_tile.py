"""Extract a real, publisher-verified 160x288 expert tile without unpacking a model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from checkpoint import Checkpoint, decode_mxfp4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    c = Checkpoint(a.model)
    names = ["model.layers.0.mlp.experts.gate_up_proj_" + part for part in ("blocks", "scales")]
    receipts = {r["file"]: r for r in json.loads((a.model / "download-receipts.json").read_text())}
    for name in names:
        if not receipts[c.index[name]]["publisher_verified"]:
            raise ValueError("Real shard has not passed publisher hash verification")
    blocks = np.array(c.tensor(names[0])[0, :160, :9, :], copy=True)
    scales = np.array(c.tensor(names[1])[0, :160, :9], copy=True)
    if not np.all((scales >= 4) & (scales <= 249)):
        raise ValueError("Tile outside the first kernel's normal FP32 scale domain")
    matrix = decode_mxfp4(blocks, scales).reshape(160, 288)
    # Independent exact special-value and nibble-order checks.
    exhaustive = np.arange(256, dtype=np.uint8).reshape(16, 16)
    decoded = decode_mxfp4(exhaustive, np.full(16, 127, np.uint8)).reshape(-1)
    magnitudes = [0., .5, 1., 1.5, 2., 3., 4., 6.]
    scalar = np.array([(-1 if n & 8 else 1) * magnitudes[n & 7]
                       for b in range(256) for n in (b & 15, b >> 4)], np.float32)
    if not np.array_equal(decoded.view(np.uint32), scalar.view(np.uint32)):
        raise ValueError("All-nibble signed-zero verification failed")
    vectors = np.stack([
        np.sin(np.arange(288, dtype=np.float64) * .17).astype(np.float32),
        ((np.arange(288) % 19 - 9) / 16).astype(np.float32),
        np.zeros(288, np.float32),
    ])
    # FP64 oracle and absolute summation-error bound, independent of CSL order.
    exact = vectors.astype(np.float64) @ matrix.astype(np.float64).T
    sums = np.abs(vectors.astype(np.float64)) @ np.abs(matrix.astype(np.float64)).T
    gamma = (2 * 288 * 2**-24) / (1 - 2 * 288 * 2**-24)
    bounds = gamma * sums + np.finfo(np.float32).tiny
    a.output.mkdir(parents=True, exist_ok=True)
    target = a.output / "fixture.npz"
    with target.open("xb") as out:
        np.savez(out, blocks=blocks, scales=scales, vectors=vectors, exact=exact, bounds=bounds)
    manifest = dict(model="openai/gpt-oss-20b", revision="6cee5e81ee83917806bbde320786a8fb61efebee",
                    layer=0, expert=0, rows=[0,160], columns=[0,288],
                    sources={n: receipts[c.index[n]] for n in names},
                    fixture_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    scale_min=int(scales.min()), scale_max=int(scales.max()),
                    packed_weight_bytes=int(blocks.nbytes+scales.nbytes),
                    all_256_packed_bytes_decode_exact=True, full_model=False,
                    numerical_policy="FP64 dot oracle; gamma(2*288)*sum(abs(products)) absolute bound")
    (a.output / "fixture.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
