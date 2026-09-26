"""Original model equations and persistent KV updates, two consecutive tokens."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from backend import runtime

def execute(runner,types,order):
    meta=json.loads(Path('transformer-fixture.json').read_text())
    assert hashlib.sha256(Path('transformer-fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
    with np.load('transformer-fixture.npz',allow_pickle=False) as source:f={k:source[k] for k in source.files}
    names=['hidden','gain','normalized','rotary_input','rotary_output','frequencies','position',
           'query','new_key','new_value','keys','values','sinks','attention','calls']
    ids={n:runner.get_id(n) for n in names}
    opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
    def put(name,data,x,bits=16):
        value=np.asarray(data,dtype=np.float32 if name=='frequencies' else np.uint32).reshape(-1)
        runner.memcpy_h2d(ids[name],value,x,0,1,1,value.size,data_type=types.MEMCPY_16BIT if bits==16 else types.MEMCPY_32BIT,**opts)
    def get(name,count,x):
        value=np.zeros(count,np.uint32)
        runner.memcpy_d2h(value,ids[name],x,0,1,1,count,data_type=types.MEMCPY_16BIT,**opts)
        return (value&65535).astype(np.uint16)
    records=[]
    def check(name,actual,expected):
        a=(actual.astype(np.uint32)<<16).view(np.float32)
        e=(expected.astype(np.uint32)<<16).view(np.float32)
        error=np.abs(a-e)
        bound=np.maximum(np.abs(e)*2**-7,np.float32(2**-133))
        passed=bool(np.isfinite(a).all() and np.all(error<=bound))
        records.append(dict(operator=name,passed=passed,exact=int(np.sum(actual==expected)),count=int(actual.size),max_abs_error=float(error.max())))
        Path('observations.json').write_text(json.dumps(records,indent=2)+'\n')
        np.save(name+'.npy',actual)
        if not passed:raise ValueError(name+' exceeds BF16 rounding tolerance')
    put('gain',f['gain'],0);put('frequencies',f['frequencies'],0,32);put('sinks',f['sinks'],1)
    for position in range(2):
        put('position',[position],0,32);put('position',[position],1,32)
        put('hidden',f['hidden'][position],0);runner.launch('normalize',nonblock=False)
        check('norm-'+str(position),get('normalized',2880,0),f['normalized'][position])
        for name in ('q','k'):
            put('rotary_input',f['raw_'+name][position],0);runner.launch('rotate',nonblock=False)
            check('rope-'+name+'-'+str(position),get('rotary_output',64,0),f['rotary_'+name][position])
        put('query',f['queries'][position],1);put('new_key',f['keys'][position],1);put('new_value',f['values'][position],1)
        runner.launch('step',nonblock=False)
        check('attention-'+str(position),get('attention',512,1),f['attention'][position])
        for name in ('keys','values'):
            cache=get(name,6144,1).reshape(96,64)
            if not np.array_equal(cache[:position+1],f[name][:position+1]) or np.any(cache[position+1:]):
                raise ValueError('Persistent KV cache mismatch: '+name)
    if not np.array_equal(get('gain',2880,0),f['gain']) or not np.array_equal(get('sinks',8,1),f['sinks']):
        raise ValueError('Parameter retention failed')
    counters=np.zeros(6,np.uint32)
    runner.memcpy_d2h(counters,ids['calls'],0,0,2,1,3,data_type=types.MEMCPY_32BIT,**opts)
    if not np.array_equal(counters,[2,4,0,0,0,2]):raise ValueError('Operator call count mismatch')
    return records

def main():
    p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');args=p.parse_args()
    with runtime(args.physical) as (runner,types,order):records=execute(runner,types,order)
    result=dict(passed=True,normal_stop=True,physical=args.physical,full_model=False,
       scope='original RMSNorm, YaRN RoPE, GQA learned-sink attention and persistent two-token KV',observations=records)
    Path('result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(records),flush=True)

if __name__=='__main__':main()
