import argparse,hashlib,json
from pathlib import Path
import numpy as np
from backend import runtime

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--physical',action='store_true');args=parser.parse_args()
    meta=json.loads(Path('fixture.json').read_text());assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
    with np.load('fixture.npz',allow_pickle=False) as src:f={k:src[k] for k in src.files}
    observations=[]
    with runtime(args.physical) as (runner,dtype,order):
        ids={n:runner.get_id(n) for n in ['input','gain','output','probes','values','calls']}
        opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
        def put(n,data,x,bits):
            runner.memcpy_h2d(ids[n],data,x,0,1,1,data.size,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
        def get(n,count,x,bits,kind):
            data=np.zeros(count,kind)
            runner.memcpy_d2h(data,ids[n],x,0,1,1,count,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
            return data
        put('gain',f['gain'].astype(np.uint32),0,16);put('probes',f['probes'].copy(),1,32)
        for i,inputs in enumerate(f['inputs']):
            put('input',inputs.astype(np.uint32),0,16);runner.launch('compute',nonblock=False)
            bits=get('output',5120,0,16,np.uint32)&65535
            output=(bits<<16).view(np.float32)
            math=get('values',512,1,32,np.float32).reshape(4,128)
            errors=np.abs(math.astype(np.float64)-f['truth'])
            put('input',inputs.astype(np.uint32),0,16);runner.launch('inplace',nonblock=False)
            inplace=get('input',5120,0,16,np.uint32)&65535
            counts=[int(get('calls',1,x,32,np.uint32)[0]) for x in range(2)]
            record=dict(case=i,norm_pass=bool(np.isfinite(output).all() and np.all(output>=f['norm_lower'][i]) and np.all(output<=f['norm_upper'][i])),
                nonlinear_pass=bool(np.isfinite(math).all() and np.all(errors<=f['bounds'])),
                inplace_pass=bool(np.array_equal(inplace,bits)),counter_pass=counts==[2*(i+1)]*2,
                nonlinear_max_error_ratio=float(np.max(errors/f['bounds'])))
            observations.append(record);Path('observations.json').write_text(json.dumps(observations,indent=2)+'\n')
            np.savez(f'actual-{i}.npz',norm=bits,nonlinear=math)
            print(json.dumps(record),flush=True)
            if not all(v for k,v in record.items() if k.endswith('_pass')):raise ValueError('Qwen math acceptance failed')
        assert np.array_equal(get('gain',5120,0,16,np.uint32)&65535,f['gain'])
    Path('result.json').write_text(json.dumps(dict(passed=True,physical=args.physical,normal_stop=True,full_model=False,
        scope=meta['scope'],gain_retention=True,observations=observations),indent=2)+'\n')

if __name__=='__main__':main()
