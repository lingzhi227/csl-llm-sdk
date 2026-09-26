"""No host numerical operations in the request; host oracle checks afterwards."""
import argparse,faulthandler,hashlib,json,time
from pathlib import Path
import numpy as np
from backend import runtime

def round_bf16(x):
    b=np.asarray(x,np.float32).copy().view(np.uint32)
    return ((b+np.uint32(0x7fff)+((b>>16)&1))&np.uint32(0xffff0000)).view(np.float32)

def main():
    faulthandler.enable();p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');args=p.parse_args()
    meta=json.loads(Path('fixture.json').read_text())
    assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
    with np.load('fixture.npz',allow_pickle=False) as src:f={k:src[k] for k in src.files}
    records=[];expected=np.zeros((5,9,6),np.uint32)
    expected[:,:,0]=13;expected[:,:,4]=1;expected[:,:,5]=13
    expected[4,8,0]=0;expected[4,8,5]=0;expected[4,8,3]=6
    for t in meta['tiles']:
        expected[t['y'],t['x'],1]=1
        if t['matrix']==3:expected[t['y'],t['x'],2]=1
    for row,source_x in [(0,0),(0,1),(1,4),(1,5),(2,3),(3,6)]:
        expected[row,source_x:,3]+=1;expected[row+1:4,8,3]+=1
    candidates=np.r_[np.arange(140,dtype=np.uint32),np.arange(248220,248320,dtype=np.uint32)]
    with runtime(args.physical) as (runner,dtype,order):
        ids={n:runner.get_id(n) for n in ['weights','controls','config','input','results','winner','trace']}
        opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
        def put(n,data,x,y,bits=32):
            runner.memcpy_h2d(ids[n],data,x,y,1,1,data.size,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
        def get(n,count,x,y,width=1,height=1,kind=np.float32,bits=32):
            data=np.zeros(count*width*height,kind)
            runner.memcpy_d2h(data,ids[n],x,y,width,height,count,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
            return data
        for index,t in enumerate(meta['tiles']):
            put('weights',f['weights'][index].astype(np.uint32),t['x'],t['y'],16)
            put('config',np.array([t['matrix'],t['k']],np.uint32),t['x'],t['y'])
        runner.launch('start',nonblock=False)
        for call,inputs in enumerate(f['inputs'],1):
            put('input',inputs.astype(np.uint32),8,4,16);put('controls',f['controls'][call-1].copy(),8,4)
            start=time.monotonic();runner.launch('run_test',nonblock=False)
            output=get('results',792,8,4);seconds=time.monotonic()-start
            trace=get('trace',6,0,0,9,5,kind=np.uint32).reshape(5,9,6);winner=get('winner',2,8,4,kind=np.uint32)
            errors=np.abs(output[512:].astype(np.float64)-f['exact'][call-1])
            logits=round_bf16(output[512:752]);pick=int(np.argmax(logits))
            lower=round_bf16(f['exact'][call-1,:240]-f['bounds'][call-1,:240])
            upper=round_bf16(f['exact'][call-1,:240]+f['bounds'][call-1,:240])
            argmax_pass=bool(winner[0]==candidates[pick] and winner[1]==logits.view(np.uint32)[pick] and upper[pick]>=lower.max())
            if call==3:argmax_pass=argmax_pass and int(winner[0])==0
            record=dict(call=call,max_abs_error=float(errors.max()),embedding_pass=bool(np.array_equal(output[:512].view(np.uint32),f['embedding'][call-1].view(np.uint32))),
                numerical_pass=bool(np.isfinite(output).all() and np.all(errors<=f['bounds'][call-1])),
                argmax_pass=argmax_pass,winner_id=int(winner[0]),trace_pass=bool(np.array_equal(trace,expected*call)),host_roundtrip_seconds=seconds)
            np.savez(f'actual-{call}.npz',output=output,trace=trace,winner=winner)
            records.append(record);Path('observations.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(record),flush=True)
            if not all(record[k] for k in ['embedding_pass','numerical_pass','argmax_pass','trace_pass']):raise ValueError('BF16 resident numerical/embedding/selection/route check failed')
        for index,t in enumerate(meta['tiles']):
            assert np.array_equal(get('weights',17920,t['x'],t['y'],kind=np.uint32,bits=16)&65535,f['weights'][index])
    Path('result.json').write_text(json.dumps(dict(passed=True,normal_stop=True,physical=args.physical,
        full_model=False,full_head=False,scope=meta['scope'],device_controlled=True,retention_pass=True,calls=records),indent=2)+'\n')

if __name__=='__main__':main()
