"""Validate all downloaded extents and finite FP8/scales with bounded RAM."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument("--model",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    complete=json.loads((a.model/"COMPLETE.json").read_text())
    receipts={v["file"]:v for v in complete["files"]}
    index=json.loads((a.model/"model.safetensors.index.json").read_text())["weight_map"]
    tensors={};records=[];payload=0
    for shard in sorted(set(index.values())):
        path=a.model/shard;pin=receipts[shard];assert pin["publisher_verified"] and path.stat().st_size==pin["bytes"]
        digest=hashlib.sha256()
        with path.open("rb") as stream:
            for part in iter(lambda:stream.read(4<<20),b""):digest.update(part)
        assert digest.hexdigest()==pin["sha256"]
        with path.open("rb") as stream:
            length=struct.unpack("<Q",stream.read(8))[0];assert length<1<<20
            header=json.loads(stream.read(length))
        end=0;nan_count=0;scales_count=0;scale_min=None;scale_max=None
        for name,s in sorted(((k,v) for k,v in header.items() if k!="__metadata__"),key=lambda v:v[1]["data_offsets"][0]):
            lo,hi=s["data_offsets"];assert lo==end and index[name]==shard and name not in tensors
            assert hi-lo==math.prod(s["shape"])*{"BF16":2,"F8_E4M3":1}[s["dtype"]]
            end=hi;payload+=hi-lo;tensors[name]=dict(s,shard=shard,payload_offset=length+8)
            if s["dtype"]=="F8_E4M3":
                mm=np.memmap(path,mode="r",dtype=np.uint8,offset=length+8+lo,shape=(hi-lo,))
                for start in range(0,len(mm),4<<20):nan_count+=int(np.count_nonzero((mm[start:start+(4<<20)]&127)==127))
                del mm
                scale=header[name+"_scale_inv"]
                assert scale["shape"]==[math.ceil(s["shape"][0]/128),math.ceil(s["shape"][1]/128)]
            elif name.endswith("weight_scale_inv"):
                mm=np.memmap(path,mode="r",dtype="<u2",offset=length+8+lo,shape=(math.prod(s["shape"]),))
                values=(np.array(mm,dtype=np.uint32)<<16).view(np.float32);del mm
                assert np.isfinite(values).all() and np.all(values>0)
                vmin,vmax=float(values.min()),float(values.max());scales_count+=values.size
                scale_min=vmin if scale_min is None else min(vmin,scale_min)
                scale_max=vmax if scale_max is None else max(vmax,scale_max)
        assert end+8+length==pin["bytes"] and nan_count==0
        records.append(dict(shard=shard,sha256=pin["sha256"],payload_bytes=end,fp8_nan_count=nan_count,
                            scales_checked=int(scales_count),scale_min=scale_min,scale_max=scale_max))
        print(json.dumps(records[-1]),flush=True)
    assert set(tensors)==set(index)
    result=dict(passed=True,repository=complete["repository"],revision=complete["revision"],
                tensor_count=len(tensors),payload_bytes=payload,shards=records,full_model_inference=False)
    with a.output.open("x") as stream:json.dump(result,stream,indent=2);stream.write("\n")


if __name__=="__main__":main()
