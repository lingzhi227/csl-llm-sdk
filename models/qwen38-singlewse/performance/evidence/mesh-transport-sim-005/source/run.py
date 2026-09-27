"""Numerical/protocol and cycle observations for the generated transport slice."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from backend import runtime


def main():
    p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');a=p.parse_args()
    plan=json.loads(Path('plan.json').read_text())['plan']
    w,h,capacity=plan['width'],plan['height'],plan['max_words'];n=w*h
    records=[]
    print(json.dumps(dict(stage='runtime_initialization')),flush=True)
    with runtime(a.physical) as (runner,dtype,order):
        ids={k:runner.get_id(k) for k in ['packet','audit','ticks']}
        opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
        def read(name,count,bits=32):
            data=np.zeros(n*count,np.uint32)
            runner.memcpy_d2h(data,ids[name],0,0,w,h,count,
                data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
            return data.reshape(h,w,count)
        print(json.dumps(dict(stage='runtime_ready')),flush=True)
        for case,(length,repeats,seed) in enumerate([(1,1,1),(min(33,capacity),64,17),(capacity,128,97),(min(33,capacity),128,41),(capacity,64,97)]):
            payload=((np.arange(capacity,dtype=np.uint32)*np.uint32(2654435761)) ^ np.uint32(seed))
            runner.memcpy_h2d(ids['packet'],payload[:length].copy(),0,0,1,1,length,data_type=dtype.MEMCPY_32BIT,**opts)
            print(json.dumps(dict(stage='launch',case=case)),flush=True)
            before=time.monotonic()
            runner.launch('benchmark',np.uint16(repeats),np.uint16(length),nonblock=False)
            wall=time.monotonic()-before
            print(json.dumps(dict(stage='readback',case=case)),flush=True)
            counts=read('audit',4);actual=read('packet',length)
            expected=payload.copy();expected[0]=repeats-1
            np.testing.assert_array_equal(actual[:,:,:length],np.broadcast_to(expected[:length],(h,w,length)))
            assert np.all(counts[:,:,0]==repeats) and np.all(counts[:,:,2]==repeats)
            for y in range(h):
                for x in range(w):
                    children=int(x+1<w)+int(x==0 and y+1<h)
                    assert counts[y,x,3]==repeats*children
            checksum=n*(n+1)//2+n*(repeats-1)
            assert int(counts[0,0,1])==checksum
            raw=read('ticks',6,16)[0,0].astype(np.uint64)
            start=sum(int(raw[i])<<(16*i) for i in range(3))
            end=sum(int(raw[i+3])<<(16*i) for i in range(3))
            cycles=(end-start)% (1<<48)
            assert 0<cycles<(1<<47)
            record=dict(case=case,payload_words=length,iterations=repeats,seed=seed,
                        root_cycles=cycles,cycles_per_roundtrip=cycles/repeats,
                        host_launch_seconds=wall,clock_hz=None,
                        packet_every_endpoint_pass=True,epoch_every_endpoint_pass=True,
                        children_every_endpoint_pass=True,root_checksum=checksum)
            records.append(record)
            print(json.dumps(record),flush=True)
    result=dict(passed=True,normal_stop=True,physical=a.physical,full_model=False,
                scope='arrival-driven 2D multicast and tree acknowledgement; no neural arithmetic',
                application=[w,h],cases=records,
                rate_is_not_model_tokens_per_second=True,
                timing='same root 48-bit cycle counter; no frequency assumption; host launch separately')
    Path('result.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
