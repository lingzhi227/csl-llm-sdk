"""Full original router acceptance with west-only hidden inputs after loading."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from backend import runtime

def execute(runner,types,order):
    meta=json.loads(Path('router-fixture.json').read_text())
    assert hashlib.sha256(Path('router-fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
    with np.load('router-fixture.npz',allow_pickle=False) as f:fixture={k:f[k] for k in f.files}
    ids={n:runner.get_id(n) for n in ('weights','bias','gain','hidden','input','logits','ids','probabilities','calls')}
    opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False);start=time.monotonic()
    def cp(name,data,x,y,w,h,n,bits=16,read=False):
        dt=types.MEMCPY_16BIT if bits==16 else types.MEMCPY_32BIT
        if read:runner.memcpy_d2h(data,ids[name],x,y,w,h,n,data_type=dt,**opts)
        else:runner.memcpy_h2d(ids[name],data,x,y,w,h,n,data_type=dt,**opts)
    weights=fixture['weights'].reshape(-1).astype(np.uint32)
    cp('weights',weights,1,0,1,30,3072)
    cp('bias',fixture['bias'].astype(np.uint32),1,29,1,1,32)
    cp('gain',fixture['gain'].astype(np.uint32),0,0,1,1,2880)
    records=[]
    for step,hidden in enumerate(fixture['hidden']):
        cp('hidden',hidden.astype(np.uint32),0,0,1,1,2880)
        runner.launch('compute',nonblock=False)
        norm=np.zeros(2880,np.float32);scatter=np.zeros(2880,np.float32)
        cp('input',norm,0,0,1,1,2880,32,True);cp('input',scatter,1,0,1,30,96,32,True)
        expected=(fixture['normalized'][step].astype(np.uint32)<<16).view(np.float32)
        checks=dict(normalization=bool(np.array_equal(norm.view(np.uint32),expected.view(np.uint32))),
                    scatter=bool(np.array_equal(scatter.view(np.uint32),expected.view(np.uint32))))
        saved={}
        for name,n in [('logits',32),('ids',4),('probabilities',4)]:
            data=np.zeros(n,np.uint32);cp(name,data,2,29,1,1,n,16,True)
            saved[name]=(data&65535).astype(np.uint16)
            checks[name]=bool(np.array_equal(saved[name],fixture[name][step]))
        counters=np.zeros(90,np.uint32);cp('calls',counters,0,0,3,30,1,32,True)
        checks['all_pes_completed']=bool(np.all(counters==step+1))
        np.savez(f'actual-{step}.npz',**saved,normalized=norm,scattered=scatter,counters=counters)
        record=dict(epoch=step+1,checks=checks,ids=saved['ids'].tolist(),seconds=time.monotonic()-start)
        records.append(record);print(json.dumps(record),flush=True)
        Path('observations.json').write_text(json.dumps(records,indent=2)+'\n')
        if not all(checks.values()):raise ValueError('Full router validation failed')
    actual=np.zeros_like(weights);cp('weights',actual,1,0,1,30,3072,16,True)
    if not np.array_equal(actual&65535,weights):raise ValueError('Router weights changed')
    for name,x,y,n in [('bias',1,29,32),('gain',0,0,2880)]:
        data=np.zeros(n,np.uint32);cp(name,data,x,y,1,1,n,16,True)
        if not np.array_equal(data&65535,fixture[name]):raise ValueError('Router '+name+' changed')
    return meta,records

def main():
    p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');a=p.parse_args()
    with runtime(a.physical) as (runner,types,order):meta,records=execute(runner,types,order)
    result=dict(passed=True,physical=a.physical,normal_stop=True,full_model=False,west_to_east=True,
                scope=meta['scope'],weights_uploaded_once=True,all_weights_retained=True,epochs=records)
    Path('result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
