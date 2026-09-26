"""Real GPT-OSS MoE operands compared against original BF16 PyTorch operators."""
import hashlib,json
from pathlib import Path
import numpy as np
import argparse
from backend import runtime

def execute(runner,MemcpyDataType,MemcpyOrder):
    meta=json.loads(Path('math-fixture.json').read_text())
    assert hashlib.sha256(Path('math-fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
    with np.load('math-fixture.npz',allow_pickle=False) as archive:f={k:archive[k] for k in archive.files}
    names={n:runner.get_id(n) for n in ['logits','ids','probabilities','interleaved','activation','experts','joined']}
    opts=dict(data_type=MemcpyDataType.MEMCPY_16BIT,streaming=False,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
    def put(name,values):
        data=np.asarray(values,dtype=np.uint32).reshape(-1)
        runner.memcpy_h2d(names[name],data,0,0,1,1,data.size,**opts)
    def get(name,n):
        out=np.zeros(n,np.uint32)
        runner.memcpy_d2h(out,names[name],0,0,1,1,n,**opts)
        return (out&65535).astype(np.uint16)
    def floats(bits):return (bits.astype(np.uint32)<<16).view(np.float32)
    records=[]
    def check(name,actual,expected):
        # BF16 one-ULP tolerance covers a rounded tie caused by FP32 reduction order.
        a,e=floats(actual),floats(expected)
        unit=np.maximum(np.abs(e)*2**-7,np.float32(2**-133))
        error=np.abs(a-e)
        passed=bool(np.isfinite(a).all() and np.all(error<=unit))
        records.append(dict(operator=name,passed=passed,exact=int(np.sum(actual==expected)),count=int(actual.size),max_abs_error=float(error.max())))
        Path('observations.json').write_text(json.dumps(records,indent=2)+'\n')
        if not passed:raise ValueError(name+' numerical comparison failed')

    for i,logits in enumerate(f['logits']):
        put('logits',logits);runner.launch('route',nonblock=False)
        ids=get('ids',4)
        if not np.array_equal(ids,f['ids'][i]):raise ValueError('Top-four expert identity mismatch')
        check('router-'+str(i),get('probabilities',4),f['probabilities'][i])
    put('interleaved',f['interleaved']);runner.launch('activate',nonblock=False)
    check('swiglu',get('activation',256),f['activation'])
    put('experts',f['experts']);put('probabilities',f['join_probabilities']);runner.launch('join',nonblock=False)
    check('join4',get('joined',160),f['joined'])
    return records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--physical',action='store_true')
    args=parser.parse_args()
    with runtime(args.physical) as (runner,data_type,order):
        records=execute(runner,data_type,order)
    result=dict(passed=True,normal_stop=True,physical=args.physical,full_model=False,
                scope='real layer-0 top-four, clamped SwiGLU and four-expert sum operators',observations=records)
    Path('result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(records),flush=True)

if __name__=='__main__':main()
