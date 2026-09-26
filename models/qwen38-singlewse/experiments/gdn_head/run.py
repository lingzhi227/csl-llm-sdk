"""Only inject BF16 projected operands; convolution, gates, norm and state stay on device."""
import argparse,faulthandler,hashlib,json,time
from pathlib import Path
import numpy as np
from backend import runtime

def main():
    faulthandler.enable()
    p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');args=p.parse_args()
    meta=json.loads(Path('fixture.json').read_text())
    assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
    with np.load('fixture.npz',allow_pickle=False) as src:f={k:src[k] for k in src.files}
    records=[];first=[];first_states=[];first_gated=[]
    with runtime(args.physical) as (runner,dtype,order):
        ids={n:runner.get_id(n) for n in ['state','packet','output','rounded','calls','input','conv_weights','gate_parameters','gain','gated','history']}
        opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
        def read(n,count,x,width=1,kind=np.float32,bits=32):
            arr=np.zeros(width*count,kind)
            runner.memcpy_d2h(arr,ids[n],x,0,width,1,count,
                data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
            return arr.reshape(width,count)
        def put(n,data):
            values=data.astype(np.uint32)
            runner.memcpy_h2d(ids[n],values,0,0,1,1,values.size,data_type=dtype.MEMCPY_16BIT,**opts)
        for name in ['conv_weights','gate_parameters','gain']:put(name,f[name])
        def state():
            raw=read('state',8192,1,width=2).reshape(2,128,64)
            return np.concatenate((raw[0],raw[1]),axis=1)
        for replay,length in [(False,96),(True,8)]:
            runner.launch('reset',nonblock=False)
            assert np.count_nonzero(state())==0
            assert np.count_nonzero(read('history',1536,0,kind=np.uint32,bits=16))==0
            assert np.count_nonzero(read('calls',1,0,width=3,kind=np.uint32))==0
            for position in range(length):
                put('input',f['inputs'][position])
                start=time.monotonic();runner.launch('compute',nonblock=False)
                y=read('output',128,0)[0];seconds=time.monotonic()-start
                current=state();calls=read('calls',1,0,width=3,kind=np.uint32)
                packets=read('packet',386,0,width=3)
                gated_bits=read('gated',128,0,kind=np.uint32,bits=16)[0]&65535
                gated=(gated_bits<<16).view(np.float32)
                history=read('history',1536,0,kind=np.uint32,bits=16)[0]&65535
                packet_error=np.abs(packets[0].astype(np.float64)-f['packet'][position])
                rounded=read('rounded',128,0,kind=np.uint32,bits=16)[0]&65535
                ybits=y.view(np.uint32);expected_round=(ybits+np.uint32(0x7fff)+((ybits>>16)&1))>>16
                se=np.abs(current.astype(np.float64)-f['state'][position])
                oe=np.abs(y.astype(np.float64)-f['output'][position])
                record=dict(position=position,replay=replay,max_state_error=float(se.max()),max_output_error=float(oe.max()),
                    state_pass=bool(np.isfinite(current).all() and np.all(se<=f['state_bound'][position])),
                    output_pass=bool(np.isfinite(y).all() and np.all(oe<=f['output_bound'][position])),
                    round_pass=bool(np.array_equal(rounded,expected_round)),
                    packet_pass=bool(np.isfinite(packets).all() and np.all(packets.view(np.uint32)==packets[0].view(np.uint32)) and np.all(packet_error<=f['packet_bound'][position])),
                    history_pass=bool(np.array_equal(history,f['history'][position].reshape(-1))),
                    gated_pass=bool(np.isfinite(gated).all() and np.all(gated>=f['gated_lower'][position]) and np.all(gated<=f['gated_upper'][position])),
                    counter_pass=bool(np.all(calls==position+1)),
                    replay_pass=bool(not replay or (np.array_equal(y,first[position]) and np.array_equal(current,first_states[position]) and np.array_equal(gated_bits,first_gated[position]))),
                    host_roundtrip_seconds=seconds)
                if not replay and position<8:first.append(y.copy());first_states.append(current.copy());first_gated.append(gated_bits.copy())
                np.savez(f'actual-{int(replay)}-{position:03d}.npz',state=current,output=y,rounded=rounded,gated=gated_bits,packet=packets,history=history)
                records.append(record)
                Path('observations.json').write_text(json.dumps(records,indent=2)+'\n')
                print(json.dumps(record),flush=True)
                if not all(v for k,v in record.items() if k.endswith('_pass')):raise ValueError('Recurrent numerical or state acceptance failed')
        for name in ['conv_weights','gate_parameters','gain']:
            assert np.array_equal(read(name,f[name].size,0,kind=np.uint32,bits=16)[0]&65535,f[name].reshape(-1))
    Path('result.json').write_text(json.dumps(dict(passed=True,normal_stop=True,physical=args.physical,full_model=False,
        scope=meta['scope'],positions=96,reset_replay_positions=8,device_state=True,device_broadcast_and_join=True,parameters_retained=True,connected_preprocess_and_gated_norm=True),indent=2)+'\n')

if __name__=='__main__':main()
