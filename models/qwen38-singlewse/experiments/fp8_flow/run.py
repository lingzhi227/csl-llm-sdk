"""Host sends FP32 activations only to west; east must receive device results."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from backend import runtime

def main():
    p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');a=p.parse_args()
    meta=json.loads(Path('fixture.json').read_text())
    assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
    with np.load('fixture.npz',allow_pickle=False) as src:f={k:src[k] for k in src.files}
    records=[]
    with runtime(a.physical) as (runner,dtype,order):
        ids={n:runner.get_id(n) for n in ['weights','scales','input','packet','output','calls']}
        opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
        def upload(n,v,x,bits=32):
            runner.memcpy_h2d(ids[n],v,x,0,1,1,v.size,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
        def read(n,count,x,width=1,bits=32,kind=np.float32):
            v=np.zeros(count*width,kind)
            runner.memcpy_d2h(v,ids[n],x,0,width,1,count,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
            return v.reshape(width,count)
        weights=f['weights'].astype(np.uint32)
        upload('weights',weights,1,16);upload('scales',f['scales'],1)
        for i,x in enumerate(f['vectors']):
            upload('input',x.copy(),0)
            start=time.monotonic();runner.launch('compute',nonblock=False)
            y=read('output',272,2)[0];seconds=time.monotonic()-start
            packets=read('packet',33,0,width=2,kind=np.uint32)
            calls=read('calls',1,0,width=3,kind=np.uint32)
            wb=read('weights',weights.size,1,bits=16,kind=np.uint32)[0]
            sb=read('scales',3,1)[0]
            error=np.abs(y.astype(np.float64)-f['exact'][i])
            record=dict(call=i+1,max_abs_error=float(error.max()),max_abs_bound=float(f['bounds'][i].max()),
                numerical_pass=bool(np.isfinite(y).all() and np.all(error<=f['bounds'][i])),
                packet_pass=bool(np.all(packets==f['packets'][i])),
                counter_pass=bool(np.all(calls==i+1)),
                retention_pass=bool(np.array_equal(wb&65535,weights) and np.array_equal(sb.view(np.uint32),f['scales'].view(np.uint32))),
                host_roundtrip_seconds=seconds)
            records.append(record);np.save(f'actual-{i}.npy',y)
            Path('observations.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(record),flush=True)
            if not all(record[k] for k in ['numerical_pass','packet_pass','counter_pass','retention_pass']):
                raise ValueError('FP8 flow acceptance failed')
    Path('result.json').write_text(json.dumps(dict(passed=True,normal_stop=True,physical=a.physical,
        full_model=False,west_to_east=True,scope='Three-PE quantization/packed transfer/FP8 block GEMV/result transfer, original 272x128 phase16 tile',calls=records),indent=2)+'\n')

if __name__=='__main__':main()
