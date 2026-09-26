"""Only inject head operands; all recurrent state and shard joins stay on device."""
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
    records=[];first=[];first_states=[]
    with runtime(args.physical) as (runner,dtype,order):
        ids={n:runner.get_id(n) for n in ['state','packet','output','rounded','calls']}
        opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
        def read(n,count,x,width=1,kind=np.float32,bits=32):
            arr=np.zeros(width*count,kind)
            runner.memcpy_d2h(arr,ids[n],x,0,width,1,count,
                data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
            return arr.reshape(width,count)
        def state():
            raw=read('state',8192,1,width=2).reshape(2,128,64)
            return np.concatenate((raw[0],raw[1]),axis=1)
        for replay,length in [(False,96),(True,8)]:
            runner.launch('reset',nonblock=False)
            assert np.count_nonzero(state())==0
            assert np.count_nonzero(read('calls',1,0,width=3,kind=np.uint32))==0
            for position in range(length):
                packet=f['packet'][position].copy()
                runner.memcpy_h2d(ids['packet'],packet,0,0,1,1,386,data_type=dtype.MEMCPY_32BIT,**opts)
                start=time.monotonic();runner.launch('compute',nonblock=False)
                y=read('output',128,0)[0];seconds=time.monotonic()-start
                current=state();calls=read('calls',1,0,width=3,kind=np.uint32)
                packets=read('packet',386,0,width=3)
                rounded=read('rounded',128,0,kind=np.uint32,bits=16)[0]&65535
                ybits=y.view(np.uint32);expected_round=(ybits+np.uint32(0x7fff)+((ybits>>16)&1))>>16
                se=np.abs(current.astype(np.float64)-f['state'][position])
                oe=np.abs(y.astype(np.float64)-f['output'][position])
                record=dict(position=position,replay=replay,max_state_error=float(se.max()),max_output_error=float(oe.max()),
                    state_pass=bool(np.isfinite(current).all() and np.all(se<=f['state_bound'][position])),
                    output_pass=bool(np.isfinite(y).all() and np.all(oe<=f['output_bound'][position])),
                    round_pass=bool(np.array_equal(rounded,expected_round)),
                    packet_pass=bool(np.all(packets.view(np.uint32)==packet.view(np.uint32))),
                    counter_pass=bool(np.all(calls==position+1)),
                    replay_pass=bool(not replay or (np.array_equal(y,first[position]) and np.array_equal(current,first_states[position]))),
                    host_roundtrip_seconds=seconds)
                if not replay and position<8:first.append(y.copy());first_states.append(current.copy())
                np.savez(f'actual-{int(replay)}-{position:03d}.npz',state=current,output=y,rounded=rounded)
                records.append(record)
                Path('observations.json').write_text(json.dumps(records,indent=2)+'\n')
                print(json.dumps(record),flush=True)
                if not all(v for k,v in record.items() if k.endswith('_pass')):raise ValueError('Recurrent numerical or state acceptance failed')
    Path('result.json').write_text(json.dumps(dict(passed=True,normal_stop=True,physical=args.physical,full_model=False,
        scope=meta['scope'],positions=96,reset_replay_positions=8,device_state=True,device_broadcast_and_join=True),indent=2)+'\n')

if __name__=='__main__':main()
