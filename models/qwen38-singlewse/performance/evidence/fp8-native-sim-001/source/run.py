"""Exhaustive finite FP8 product test through scaled native FP16 inputs."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from backend import runtime


def decode(c):
    sign=-1 if c&128 else 1;exponent=(c>>3)&15;mantissa=c&7
    return sign*(mantissa*2.0**-9 if exponent==0 else (1+mantissa/8)*2.0**(exponent-7))


def main():
    p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');a=p.parse_args()
    codes=[c for c in range(256) if c&127!=127]
    bits=np.array([((c&127)<<7)|((c&128)<<8) for c in codes],np.uint32)
    exact=np.array([decode(c) for c in codes],np.float64)
    records=[];bad=[];all_start=time.monotonic()
    with runtime(a.physical) as (runner,dtype,order):
        ids={n:runner.get_id(n) for n in ['weights','input','output','ticks','calls']}
        opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
        runner.memcpy_h2d(ids['weights'],bits,0,0,1,1,254,data_type=dtype.MEMCPY_16BIT,**opts)
        for i,(code,repeats) in enumerate([(c,1) for c in codes]+[(1,128),(126,128)]):
            value=np.array([((code&127)<<7)|((code&128)<<8)],np.uint32)
            runner.memcpy_h2d(ids['input'],value,0,0,1,1,1,data_type=dtype.MEMCPY_16BIT,**opts)
            runner.launch('compute',np.uint16(repeats),nonblock=False)
            actual=np.zeros(254,np.float32)
            runner.memcpy_d2h(actual,ids['output'],0,0,1,1,254,data_type=dtype.MEMCPY_32BIT,**opts)
            expected=(exact*decode(code)*(repeats/65536)).astype(np.float32)
            # +0 accumulator canonicalizes a zero product; require exact finite values.
            mismatch=np.flatnonzero(~np.isfinite(actual)|(actual!=expected))
            if mismatch.size:
                for j in mismatch[:8]:bad.append(dict(input_code=code,weight_code=codes[j],repeats=repeats,actual=float(actual[j]),expected=float(expected[j])))
                break
            if repeats==128:
                ticks=np.zeros(6,np.uint32)
                runner.memcpy_d2h(ticks,ids['ticks'],0,0,1,1,6,data_type=dtype.MEMCPY_16BIT,**opts)
                start=sum(int(ticks[k])<<(16*k) for k in range(3));end=sum(int(ticks[k+3])<<(16*k) for k in range(3))
                cycles=(end-start)%(1<<48);assert 0<cycles<(1<<47)
                records.append(dict(input_code=code,repeats=repeats,elements=254,cycles=cycles,cycles_per_fma_vector=cycles/repeats,physical=a.physical,clock_hz=None))
            if i%32==0:print(json.dumps(dict(cases_completed=i+1,elapsed_seconds=time.monotonic()-all_start)),flush=True)
        wb=np.zeros(254,np.uint32);runner.memcpy_d2h(wb,ids['weights'],0,0,1,1,254,data_type=dtype.MEMCPY_16BIT,**opts)
        assert np.array_equal(wb,bits)
    result=dict(passed=not bad,normal_stop=True,physical=a.physical,full_model=False,
                scope='native mixed FP16 multiply / FP32 accumulate using exact scaled FP8 encodings, no packed decode or model inference',
                completed_cases=i+1,exhaustive_ordered_products=64516 if i>=253 and not bad else None,
                failures=bad,timing=records,weights_retained=True,
                fp32_comparison='exact finite equality; signed zero canonicalized by positive-zero FMA accumulator')
    Path('result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if bad:raise ValueError('Native FP16 semantics invalidate the candidate transform')

if __name__=='__main__':main()
