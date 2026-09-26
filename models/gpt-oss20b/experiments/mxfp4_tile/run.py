"""Compare three real-weight calls and retention in a single device runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def execute(runner, data_type, order):
    config = json.loads(Path("experiment.json").read_text()) if Path("experiment.json").exists() else {}
    flow = config.get("west_to_east", False)
    weight_x, output_x = (1, 2) if flow else (0, 0)
    meta = json.loads(Path("fixture.json").read_text())
    if hashlib.sha256(Path("fixture.npz").read_bytes()).hexdigest() != meta["fixture_sha256"]:
        raise ValueError("Fixture hash mismatch")
    with np.load("fixture.npz", allow_pickle=False) as source:
        f = {k: source[k] for k in source.files}
    weights = f["blocks"].reshape(-1).view("<u2").astype(np.uint32)
    scales = f["scales"].reshape(-1).view("<u2").astype(np.uint32)
    ids = {n: runner.get_id(n) for n in ("weights", "scales", "input", "output", "calls")}
    opts = dict(streaming=False, order=order.ROW_MAJOR, nonblock=False)
    def node(name):
        return output_x if name == "output" else weight_x if name in ("weights", "scales", "received_input") else 0
    def upload(name, value, bits=32):
        runner.memcpy_h2d(ids[name], value, node(name), 0, 1, 1, value.size,
                          data_type=data_type.MEMCPY_16BIT if bits==16 else data_type.MEMCPY_32BIT, **opts)
    def download(name, count, bits=32, dtype=np.float32):
        out = np.zeros(count, dtype=dtype)
        runner.memcpy_d2h(out, ids[name], node(name), 0, 1, 1, count,
                          data_type=data_type.MEMCPY_16BIT if bits==16 else data_type.MEMCPY_32BIT, **opts)
        return out
    upload("weights", weights, 16)
    upload("scales", scales, 16)
    records = []
    for i, vector in enumerate(f["vectors"]):
        upload("input", vector.copy())
        start = time.monotonic()
        runner.launch("compute", nonblock=False)
        elapsed = time.monotonic()-start
        y = download("output", 160)
        count = download("calls", 1, dtype=np.uint32)
        wb = download("weights", weights.size, 16, np.uint32)
        sb = download("scales", scales.size, 16, np.uint32)
        error = np.abs(y.astype(np.float64)-f["exact"][i])
        numerical = bool(np.isfinite(y).all() and np.all(error <= f["bounds"][i]))
        retention = bool(np.array_equal(wb & 65535, weights) and np.array_equal(sb & 65535, scales))
        if flow:
            received = np.zeros(288, np.float32)
            runner.memcpy_d2h(received, ids["input"], 1, 0, 1, 1, 288,
                              data_type=data_type.MEMCPY_32BIT, **opts)
            counters = np.zeros(3, np.uint32)
            runner.memcpy_d2h(counters, ids["calls"], 0, 0, 3, 1, 1,
                              data_type=data_type.MEMCPY_32BIT, **opts)
            if not np.array_equal(received.view(np.uint32), vector.view(np.uint32)) or not np.all(counters == i+1):
                raise ValueError("West-to-east operands or completion epochs mismatch")
        record = dict(call=i+1, numerical_pass=numerical, retained_weights=retention,
                      counter_pass=int(count[0])==i+1, max_abs_error=float(error.max()),
                      max_abs_bound=float(f["bounds"][i].max()), host_launch_seconds=elapsed)
        records.append(record)
        np.save(f"actual-{i}.npy", y)
        Path("observations.json").write_text(json.dumps(records, indent=2)+"\n")
        print(json.dumps(record), flush=True)
        if not (numerical and retention and record["counter_pass"]):
            raise ValueError("Numerical or persistent-weight check failed")
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--physical", action="store_true")
    args = p.parse_args()
    if args.physical:
        from job_capture import install
        install("run.jobs")
        from cerebras.sdk.client import SdkRuntime
        from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder
        artifact = json.loads(Path("artifact.json").read_text())
        path = Path(artifact["artifact"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError("Compiled artifact mismatch")
        with SdkRuntime(str(path), simulator=False, disable_version_check=True,
                        resource_cpu=2000, resource_mem=4<<30) as runner:
            records = execute(runner, MemcpyDataType, MemcpyOrder)
    else:
        from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder, SimfabConfig, SdkTarget, get_platform
        runner = SdkRuntime("out", get_platform(None, SimfabConfig(suppress_trace=True, num_threads=1, dump_core=False), SdkTarget.WSE3))
        runner.load()
        runner.run()
        try:
            records = execute(runner, MemcpyDataType, MemcpyOrder)
        finally:
            runner.stop()
    config = json.loads(Path("experiment.json").read_text()) if Path("experiment.json").exists() else {}
    Path("result.json").write_text(json.dumps(dict(passed=True, normal_stop=True,
        physical=args.physical, scope="one original 160x288 MXFP4 expert tile, three calls",
        full_model=False, west_to_east=config.get("west_to_east",False), calls=records), indent=2)+"\n")


if __name__ == "__main__":
    main()
