"""Resident expert acceptance; host only uploads inputs after initialization."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from backend import runtime

def compare(actual,expected):
    a=(actual.astype(np.uint32)<<16).view(np.float32)
    b=(expected.astype(np.uint32)<<16).view(np.float32)
    # Monotone signed encoding: +0 and -0 share coordinate zero.
    def ordered(x):
        x=x.astype(np.int32)
        return np.where(x&32768,-(x&32767),x)
    ulp=np.abs(ordered(actual)-ordered(expected))
    return dict(passed=bool(np.isfinite(a).all() and np.isfinite(b).all() and (ulp<=1).all()),
                max_bf16_ulp=int(ulp.max()),bit_exact=int(np.count_nonzero(actual==expected)),
                values=int(actual.size),max_abs_error=float(np.max(np.abs(a-b))))

def execute(runner,types,order):
    meta=json.loads(Path('expert-fixture.json').read_text());gr=meta['gate_rows'];dr=meta['down_rows']
    assert hashlib.sha256(Path('expert-fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
    with np.load('expert-fixture.npz',allow_pickle=False) as f:fixture={k:f[k] for k in f.files}
    ids={n:runner.get_id(n) for n in ('weights','scales','bias','input','output','gate_debug','activation_debug','counters')}
    opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
    start=time.monotonic()
    def log(phase,**data):print(json.dumps(dict(phase=phase,seconds=time.monotonic()-start,**data)),flush=True)
    def copy(name,data,x,y,w,h,n,bits,read=False):
        dt=types.MEMCPY_16BIT if bits==16 else types.MEMCPY_32BIT
        if read:runner.memcpy_d2h(data,ids[name],x,y,w,h,n,data_type=dt,**opts)
        else:runner.memcpy_h2d(ids[name],data,x,y,w,h,n,data_type=dt,**opts)
    def bank(kind,rows,x,verify=False):
        for offset in range(0,rows,2):
            height=min(2,rows-offset)
            for field,symbol,count in [('blocks','weights',11520),('scales','scales',720)]:
                raw=fixture[kind+'_'+field][offset:offset+height]
                slots=raw.reshape(-1).view('<u2').astype(np.uint32)
                assert slots.nbytes<=1<<20
                if verify:
                    actual=np.zeros_like(slots);copy(symbol,actual,x,offset,10,height,count,16,True)
                    if not np.array_equal(actual&65535,slots):raise ValueError('Resident '+kind+' '+field+' changed')
                else:copy(symbol,slots,x,offset,10,height,count,16)
            bias=fixture[kind+'_bias'][offset:offset+height].reshape(-1).astype(np.uint32)
            if verify:
                actual=np.zeros_like(bias);copy('bias',actual,x+9,offset,1,height,160,16,True)
                if not np.array_equal(actual&65535,bias):raise ValueError('Resident bias changed')
            else:copy('bias',bias,x+9,offset,1,height,160,16)
    bank('gate',gr,1);bank('down',dr,12);log('weights_uploaded_once')
    records=[]
    for epoch,vector in enumerate(fixture['vectors'],1):
        copy('input',vector.copy(),0,0,1,1,2880,32)
        runner.launch('compute',nonblock=False);log('epoch_dispatched',epoch=epoch)
        values={}
        for symbol,x,count,key in [('gate_debug',11,gr*160,'gate'),('activation_debug',11,2880,'activation'),('output',22,dr*160,'output')]:
            data=np.zeros(count,np.uint32);copy(symbol,data,x,0,1,1,count,16,True)
            values[key]=(data&65535).astype(np.uint16)
        counters=np.zeros(gr*23*5,np.uint32);copy('counters',counters,0,0,23,gr,5,32,True)
        counters=counters.reshape(gr,23,5)
        counts_ok=bool(np.all(counters[:,:,0]==epoch) and np.all(counters[:,:,4]==epoch))
        checks={k:compare(v,fixture['expected_'+k][epoch-1]) for k,v in values.items()}
        record=dict(epoch=epoch,checks=checks,every_pe_completed=counts_ok)
        records.append(record)
        np.savez(f'actual-{epoch}.npz',**values,counters=counters)
        Path('observations.json').write_text(json.dumps(records,indent=2)+'\n');log('epoch_compared',**record)
        if not counts_ok or not all(c['passed'] for c in checks.values()):raise ValueError('Expert numerical/protocol acceptance failed')
    bank('gate',gr,1,True);bank('down',dr,12,True);log('all_resident_weights_and_biases_retained')
    return meta,records

def main():
    p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');a=p.parse_args()
    with runtime(a.physical) as (runner,types,order):meta,records=execute(runner,types,order)
    result=dict(passed=True,physical=a.physical,normal_stop=True,full_model=False,full_expert=meta['full_expert'],
                west_to_east=True,weights_uploaded_once=True,all_weights_retained=True,scope=meta['scope'],epochs=records)
    Path('result.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
