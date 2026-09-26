"""Continuous on-device KV retention, six query heads and clean request reset."""
import argparse,faulthandler,hashlib,json,time
from pathlib import Path
import numpy as np
from backend import runtime

def expanded(x):return (x.astype(np.uint32)<<np.uint32(16)).view(np.float32)
def inside(actual,lo,hi):return bool(np.isfinite(actual).all() and np.all(actual>=lo) and np.all(actual<=hi))

def main():
    faulthandler.enable();p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');args=p.parse_args()
    meta=json.loads(Path('fixture.json').read_text())
    assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
    with np.load('fixture.npz',allow_pickle=False) as src:f={k:src[k] for k in src.files}
    observations=[];first=[]
    lengths=[96,4] if args.physical else [1,1]
    with runtime(args.physical) as (runner,dtype,order):
        ids={n:runner.get_id(n) for n in ['input','gain','frequencies','output','keys','values','trace']}
        opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
        def put(name,data,bits=16):
            runner.memcpy_h2d(ids[name],data,0,0,1,1,data.size,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
        def get(name,count,x=0,width=1,bits=16,kind=np.uint32):
            data=np.zeros(width*count,kind)
            runner.memcpy_d2h(data,ids[name],x,0,width,1,count,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
            if bits==16:data&=65535
            return data.reshape(width,count)
        def cache(name,length):
            return get(name,length*64,1,4).reshape(4,length,64).transpose(1,0,2).reshape(length,256)
        put('gain',f['gain'].astype(np.uint32));put('frequencies',f['frequencies'].copy(),32)
        runner.launch('start',nonblock=False)
        for replay,length in enumerate(lengths):
            runner.launch('reset',nonblock=False)
            assert not np.count_nonzero(get('trace',6,0,5,bits=32))
            assert not np.count_nonzero(cache('keys',1)) and not np.count_nonzero(cache('values',1))
            for position in range(length):
                put('input',f['inputs'][position].astype(np.uint32))
                start=time.monotonic();runner.launch('compute',nonblock=False)
                output=get('output',1536)[0];seconds=time.monotonic()-start
                processed=get('input',3584)[0];trace=get('trace',6,0,5,bits=32)
                keys=cache('keys',position+1);values=cache('values',position+1)
                pre=expanded(np.r_[processed[:256],processed[512:2048]])
                expected=np.array([position+1,position+1,6*(position+1),6*(position+1),14*(position+1),position+1],np.uint32)
                result=dict(position=position,replay=bool(replay),
                    preprocess_pass=inside(pre,f['preprocess_lower'][position],f['preprocess_upper'][position]),
                    unchanged_input_pass=bool(np.array_equal(processed[256:512],f['inputs'][position,256:512]) and np.array_equal(processed[2048:],f['inputs'][position,2048:])),
                    key_pass=inside(expanded(keys),f['key_lower'][:position+1],f['key_upper'][:position+1]),
                    value_pass=bool(np.array_equal(values,f['value_bits'][:position+1])),
                    output_pass=inside(expanded(output).reshape(6,256),f['output_lower'][position],f['output_upper'][position]),
                    trace_pass=bool(np.all(trace==expected)),
                    replay_pass=bool(not replay or np.array_equal(output,first[position])),host_roundtrip_seconds=seconds)
                if not replay and position<lengths[1]:first.append(output.copy())
                np.savez(f'actual-{replay}-{position:03d}.npz',output=output,processed=processed,keys=keys,values=values,trace=trace)
                observations.append(result);Path('observations.json').write_text(json.dumps(observations,indent=2)+'\n');print(json.dumps(result),flush=True)
                if not all(v for k,v in result.items() if k.endswith('_pass')):raise ValueError('Attention group numerical/cache/route/reset check failed')
        if args.physical:
            assert not np.count_nonzero(cache('keys',96)[lengths[1]:])
            assert not np.count_nonzero(cache('values',96)[lengths[1]:])
        assert np.array_equal(get('gain',512)[0],f['gain'])
        assert np.array_equal(get('frequencies',32,bits=32,kind=np.float32)[0].view(np.uint32),f['frequencies'].view(np.uint32))
    Path('result.json').write_text(json.dumps(dict(passed=True,normal_stop=True,physical=args.physical,full_model=False,
        scope=meta['scope'],positions=lengths[0],reset_replay_positions=lengths[1],full_context_exercised=args.physical,
        simulator_scope='One-position smoke and reset only' if not args.physical else None,
        all_six_query_heads=True,parameters_retained=True),indent=2)+'\n')

if __name__=='__main__':main()
