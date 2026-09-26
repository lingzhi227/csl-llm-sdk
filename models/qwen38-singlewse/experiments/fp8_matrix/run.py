"""Complete matrix projection; host touches inputs only on quantizer row."""
import argparse,faulthandler,hashlib,json,time
from pathlib import Path
import numpy as np
from backend import runtime

def main():
    faulthandler.enable()
    p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');a=p.parse_args()
    meta=json.loads(Path('fixture.json').read_text())
    assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
    with np.load('fixture.npz',allow_pickle=False) as src:f={k:src[k] for k in src.files}
    records=[]
    print(json.dumps(dict(stage='before_runtime_initialization')),flush=True)
    with runtime(a.physical) as (runner,dtype,order):
        print(json.dumps(dict(stage='runtime_initialized')),flush=True)
        ids={n:runner.get_id(n) for n in ['weights','scales','input','packet','output','calls']}
        opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
        def upload(n,v,x,y,width,height,count,bits=32):
            runner.memcpy_h2d(ids[n],v,x,y,width,height,count,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
        def read(n,count,x,y,width,height,bits=32,kind=np.float32):
            v=np.zeros(count*width*height,kind)
            runner.memcpy_d2h(v,ids[n],x,y,width,height,count,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
            return v.reshape(height,width,count)
        # Bound SDK native/protobuf buffers to < 16 MiB of host words per copy.
        # All weights are still loaded once, before the first compute call.
        for first in range(0,64,4):
            chunk=f['weights'][first:first+4].astype(np.uint32).reshape(-1)
            assert chunk.nbytes<16<<20
            print(json.dumps(dict(stage='weight_upload_start',first_row=first)),flush=True)
            upload('weights',chunk,0,first+1,40,4,17408,16)
            print(json.dumps(dict(stage='weight_upload_complete',rows=first+4)),flush=True)
        upload('scales',f['scales'].reshape(-1),0,1,40,64,3)
        for i,x in enumerate(f['vectors']):
            upload('input',x.copy(),0,0,40,1,128)
            begin=time.monotonic();runner.launch('compute',nonblock=False)
            y=read('output',272,39,1,1,64).reshape(-1)
            elapsed=time.monotonic()-begin
            packets=read('packet',33,0,0,40,65,kind=np.uint32)
            counters=read('calls',1,0,0,40,65,kind=np.uint32)
            error=np.abs(y.astype(np.float64)-f['exact'][i])
            record=dict(call=i+1,max_abs_error=float(error.max()),max_abs_bound=float(f['bounds'][i].max()),
                numerical_pass=bool(np.isfinite(y).all() and np.all(error<=f['bounds'][i])),
                broadcast_pass=bool(np.all(packets==f['packets'][i])),
                counter_pass=bool(np.all(counters==i+1)),host_roundtrip_seconds=elapsed)
            records.append(record);np.save(f'actual-{i}.npy',y)
            Path('observations.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(record),flush=True)
            if not all(record[k] for k in ['numerical_pass','broadcast_pass','counter_pass']):raise ValueError('Full matrix failed')
        retention=True
        for first in range(0,64,4):
            wb=read('weights',17408,0,first+1,40,4,bits=16,kind=np.uint32)
            retention=retention and np.array_equal(wb&65535,f['weights'][first:first+4])
        sb=read('scales',3,0,1,40,64)
        retention=bool(retention and np.array_equal(sb.view(np.uint32),f['scales'].view(np.uint32)))
        if not retention:raise ValueError('Resident matrix changed')
    Path('result.json').write_text(json.dumps(dict(passed=True,normal_stop=True,physical=a.physical,
        full_model=False,full_matrix=True,west_to_east=True,retention_pass=retention,
        scope='Complete original 17408x5120 layer0 gate projection; device group128 quantization, 64-row broadcast and 40-block reductions',calls=records),indent=2)+'\n')

if __name__=='__main__':main()
