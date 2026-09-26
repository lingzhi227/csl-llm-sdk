"""Load once, then only one controller input and one request launch per case."""
import argparse,faulthandler,hashlib,json,time
from pathlib import Path
import numpy as np
from backend import runtime

def main():
    faulthandler.enable();p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');args=p.parse_args()
    meta=json.loads(Path('fixture.json').read_text())
    assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
    with np.load('fixture.npz',allow_pickle=False) as src:f={k:src[k] for k in src.files}
    records=[];expected=np.zeros((5,9,6),np.uint32)
    expected[:,:,0]=12;expected[:,:,4]=1;expected[:,:,5]=12
    expected[4,8,0]=0;expected[4,8,5]=0;expected[4,8,3]=4
    for tile in meta['tiles']:
        expected[tile['y'],tile['x'],1:3]=1
    for row,source_x in enumerate([1,5,3,6]):
        expected[row,source_x:,3]+=1
        expected[row+1:4,8,3]+=1
    with runtime(args.physical) as (runner,dtype,order):
        ids={n:runner.get_id(n) for n in ['weights','scales','config','input','results','trace']}
        opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
        def put(n,data,x,y,bits=32):
            runner.memcpy_h2d(ids[n],data,x,y,1,1,data.size,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
        def get(n,count,x,y,width=1,height=1,kind=np.float32,bits=32):
            data=np.zeros(count*width*height,kind)
            runner.memcpy_d2h(data,ids[n],x,y,width,height,count,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
            return data
        for index,tile in enumerate(meta['tiles']):
            x,y=tile['x'],tile['y']
            put('weights',f['weights'][index].astype(np.uint32),x,y,16)
            put('scales',f['scales'][index].copy(),x,y)
            put('config',np.array([tile['matrix'],tile['k']],np.uint32),x,y)
        runner.launch('start',nonblock=False)
        for call,inputs in enumerate(f['inputs'],1):
            put('input',inputs.copy(),8,4)
            start=time.monotonic();runner.launch('run_test',nonblock=False)
            output=get('results',1088,8,4);seconds=time.monotonic()-start
            trace=get('trace',6,0,0,9,5,kind=np.uint32).reshape(5,9,6)
            errors=np.abs(output.astype(np.float64)-f['exact'][call-1])
            record=dict(call=call,max_abs_error=float(errors.max()),
                numerical_pass=bool(np.isfinite(output).all() and np.all(errors<=f['bounds'][call-1])),
                trace_pass=bool(np.array_equal(trace,expected*call)),host_roundtrip_seconds=seconds)
            np.savez(f'actual-{call}.npz',output=output,trace=trace)
            records.append(record);Path('observations.json').write_text(json.dumps(records,indent=2)+'\n')
            print(json.dumps(record),flush=True)
            if not record['numerical_pass'] or not record['trace_pass']:raise ValueError('Resident bus numerical/route/ordering check failed')
        for index,tile in enumerate(meta['tiles']):
            x,y=tile['x'],tile['y']
            observed=get('weights',17408,x,y,kind=np.uint32,bits=16)&65535
            assert np.array_equal(observed,f['weights'][index])
            assert np.array_equal(get('scales',3,x,y).view(np.uint32),f['scales'][index].view(np.uint32))
    Path('result.json').write_text(json.dumps(dict(passed=True,normal_stop=True,physical=args.physical,
        full_model=False,full_matrix=False,scope=meta['scope'],device_controlled=True,retention_pass=True,calls=records),indent=2)+'\n')

if __name__=='__main__':main()
