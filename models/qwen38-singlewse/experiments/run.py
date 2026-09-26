"""Same independent numerical and retention checks in simulation and on CS3."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from backend import runtime


def main():
    p = argparse.ArgumentParser(); p.add_argument("--physical", action="store_true"); a = p.parse_args()
    meta = json.loads(Path("fixture.json").read_text())
    assert hashlib.sha256(Path("fixture.npz").read_bytes()).hexdigest() == meta["fixture_sha256"]
    with np.load("fixture.npz", allow_pickle=False) as source:
        f = {k: source[k] for k in source.files}
    records = []
    with runtime(a.physical) as (runner, dtype, order):
        ids = {n: runner.get_id(n) for n in ["weights","scales","input","output","decoded","calls"]}
        opts = dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
        def upload(n,v,bits=32):
            runner.memcpy_h2d(ids[n],v,0,0,1,1,v.size,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
        def read(n,count,bits=32,kind=np.float32):
            out=np.zeros(count,kind)
            runner.memcpy_d2h(out,ids[n],0,0,1,1,count,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
            return out
        runner.launch("decode_all",nonblock=False)
        codes=read("decoded",256)
        weights=f["weights"].astype(np.uint32)
        upload("weights",weights,16); upload("scales",f["scales"])
        for i,x in enumerate(f["vectors"]):
            upload("input",x.copy()); start=time.monotonic(); runner.launch("compute",nonblock=False)
            elapsed=time.monotonic()-start
            y=read("output",272); count=read("calls",1,kind=np.uint32)
            wb=read("weights",weights.size,16,np.uint32); sb=read("scales",3)
            error=np.abs(y.astype(np.float64)-f["exact"][i]); finite=f["finite"]
            record=dict(call=i+1,max_abs_error=float(error.max()),max_abs_bound=float(f["bounds"][i].max()),
                        numerical_pass=bool(np.isfinite(y).all() and np.all(error<=f["bounds"][i])),
                        decode_pass=bool(np.array_equal(codes[finite].view(np.uint32),f["decoded"][finite].view(np.uint32))),
                        retention_pass=bool(np.array_equal(wb&65535,weights) and np.array_equal(sb.view(np.uint32),f["scales"].view(np.uint32))),
                        counter_pass=int(count[0])==i+1,host_launch_seconds=elapsed)
            records.append(record); np.save(f"actual-{i}.npy",y)
            Path("observations.json").write_text(json.dumps(records,indent=2)+"\n"); print(json.dumps(record),flush=True)
            if not all(record[k] for k in ["numerical_pass","decode_pass","retention_pass","counter_pass"]):
                raise ValueError("FP8 tile acceptance failed")
    Path("result.json").write_text(json.dumps(dict(passed=True,normal_stop=True,physical=a.physical,
        full_model=False,scope="Original Qwen3.8 FP8 272x128 gate tile; 254 finite encodings; three calls",calls=records),indent=2)+"\n")


if __name__ == "__main__": main()
