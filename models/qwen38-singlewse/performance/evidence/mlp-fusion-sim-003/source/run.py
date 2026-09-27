"""Bit-exact fused-interface unit qualification; deliberately no timing claim."""
import hashlib
import json
from pathlib import Path
import numpy as np
from backend import runtime
meta=json.loads(Path('fixture.json').read_text())
assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
with np.load('fixture.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
captured=[]
with runtime(False) as (runner,dtype,order):
    ids={k:runner.get_id(k) for k in ['gate','up','down','original','gain','output']}
    options=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False,data_type=dtype.MEMCPY_32BIT)
    for case in range(3):
        for name in ['gate','up','down','original','gain']:
            values=np.ascontiguousarray(data[name][case]).view(np.uint32)
            runner.memcpy_h2d(ids[name],values,0,0,1,1,len(values),**options)
        runner.launch('execute',np.uint16(case),nonblock=False)
        actual=np.zeros(450,np.uint32);runner.memcpy_d2h(actual,ids['output'],0,0,1,1,450,**options)
        # Preserve failing arrays before an assertion, unlike earlier P18 probe.
        captured.append(actual);np.savez('actual.npz',actual=np.array(captured),expected=data['expected'][:case+1])
        mismatch=np.flatnonzero(actual!=data['expected'][case])
        if len(mismatch):
            Path('diagnostic.json').write_text(json.dumps(dict(case=case,mismatch_indices=mismatch.tolist(),
                actual=actual[mismatch].tolist(),expected=data['expected'][case][mismatch].tolist()))+'\n')
        np.testing.assert_array_equal(actual,data['expected'][case])
result=dict(passed=True,normal_stop=True,physical=False,full_model=False,words_checked=1350,
            fixture_sha256=meta['fixture_sha256'],scope=meta['scope'],measured_speedup=None,
            scale_implementation=meta['scale_implementation'],
            literal_division_differing_indices=[np.flatnonzero(a!=b).tolist() for a,b in zip(captured,data['literal_division'])])
Path('result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
