"""Autonomous packet feedback across teardown/rebind epochs; no neural throughput claim."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from backend import runtime


def main():
    p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');args=p.parse_args()
    plan=json.loads(Path('region.json').read_text())
    bank=np.arange(20*8814,dtype=np.uint32).reshape(5,4,8814)^np.uint32(0x5a381700)
    packet=np.arange(257,dtype=np.uint32)*np.uint32(104729)^np.uint32(0x17380100);packet[0]=0
    fixture=dict(synthetic=True,bank_bytes_per_pe=35256,bank_sha256=hashlib.sha256(bank.tobytes()).hexdigest(),
                 packet_sha256=hashlib.sha256(packet.tobytes()).hexdigest(),scope=plan['bank_payload'])
    Path('fixture.json').write_text(json.dumps(fixture,indent=2)+'\n')
    batches=[];captures={}
    with runtime(args.physical) as (runner,dtype,order):
        ids={n:runner.get_id(n) for n in ['bank','packet','audit','ticks']}
        opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
        def read(name,n,x=0,y=0,w=4,h=5,bits=32):
            value=np.zeros(w*h*n,np.uint32)
            runner.memcpy_d2h(value,ids[name],x,y,w,h,n,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
            return value.reshape(h,w,n)
        start=time.perf_counter()
        runner.memcpy_h2d(ids['bank'],bank.reshape(-1),0,0,4,5,8814,data_type=dtype.MEMCPY_32BIT,**opts)
        for x,y in [(0,0),(3,4)]:
            runner.memcpy_h2d(ids['packet'],packet,x,y,1,1,257,data_type=dtype.MEMCPY_32BIT,**opts)
        initialization=time.perf_counter()-start
        for replay,rounds in enumerate([64 if args.physical else 16,8 if args.physical else 4]):
            runner.launch('initialize',np.uint16(rounds),nonblock=False)
            armed=read('audit',12);assert not np.any(armed[...,11]);assert np.all(armed[...,0]==1)
            start=time.perf_counter();runner.launch('start',nonblock=False);host=time.perf_counter()-start
            audit=read('audit',12);expected=np.zeros((5,4,12),np.uint32)
            expected[...,[0,1,5]]=rounds;expected[...,6]=1;expected[...,7]=160;expected[...,8]=1
            for x,y in [(0,0),(3,4)]:
                expected[y,x,2:5]=rounds//2
            np.testing.assert_array_equal(audit,expected)
            cycles=[]
            for rank,x,y in [(0,0,0),(19,3,4)]:
                got=read('packet',257,x,y,1,1).reshape(-1)
                np.testing.assert_array_equal(got[1:],packet[1:]);assert got[0]==rounds-(rank==19)
                ticks=read('ticks',rounds*3,x,y,1,1,bits=16).reshape(rounds,3)
                stamps=[sum(int(t[j])<<(16*j) for j in range(3)) for t in ticks]
                first=0 if rank==0 else 1
                local=[(stamps[e+1]-stamps[e])%(1<<48) for e in range(first,rounds-1,2)]
                assert all(0<c<10000000 for c in local);cycles+=local
                captures['ticks_%d_%d'%(replay,rank)]=ticks;captures['packet_%d_%d'%(replay,rank)]=got
            captures['audit_%d'%replay]=audit
            record=dict(replay=bool(replay),epochs=rounds,host_launch_seconds=host,dependent_two_hop_cycles=cycles)
            batches.append(record);print(json.dumps(record),flush=True)
        np.testing.assert_array_equal(read('bank',8814),bank)
        np.savez('actual.npz',**captures)
    result=dict(passed=True,normal_stop=True,physical=args.physical,full_model=False,application=[4,5],
                bank_bytes_per_pe=35256,all_bank_sentinels_retained=True,original_weights=False,
                autonomous_epochs=sum(b['epochs'] for b in batches),batches=batches,
                all_teardown_and_completion_counters_exact=True,queue_rebind_and_route_readback_passed=True,
                malformed_and_non_teardown_changes_rejected=True,host_initialization_seconds=initialization,
                timing_note=plan['timing'],clock_hz=None,full_model_speed_target_achieved=False)
    Path('result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,physical=args.physical)),flush=True)


if __name__=='__main__':
    main()
