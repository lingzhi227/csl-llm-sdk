"""End-to-end full attention test; only original input crosses H2D after loading."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from backend import runtime

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def compare(actual,expected):
    a=(actual.astype(np.uint32)<<16).view(np.float32);b=(expected.astype(np.uint32)<<16).view(np.float32)
    def ordered(x):
        x=x.astype(np.int32);return np.where(x&32768,-(x&32767),x)
    ulp=np.abs(ordered(actual)-ordered(expected))
    return dict(passed=bool(np.isfinite(a).all() and np.isfinite(b).all() and (ulp<=1).all()),
      max_bf16_ulp=int(ulp.max()),bit_exact=int(np.count_nonzero(actual==expected)),values=int(actual.size),
      max_abs_error=float(np.max(np.abs(a-b))))

def execute(runner,types,order):
    meta=json.loads(Path('attention-fixture.json').read_text());assert meta['full_attention'] and meta['layer']==0
    assert digest(Path('attention-fixture.npz'))==meta['fixture_sha256']
    with np.load('attention-fixture.npz',allow_pickle=False) as f:fixture={k:f[k] for k in f.files}
    names=('identity','weights','bias','gain','frequencies','sinks','hidden','counters','keys','values')
    ids={n:runner.get_id(n) for n in names};opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
    start=time.monotonic()
    def log(phase,**data):print(json.dumps(dict(phase=phase,seconds=time.monotonic()-start,**data)),flush=True)
    def copy(name,data,x,y,w,h,n,bits,read=False):
        assert data.dtype==np.uint32 and data.flags.c_contiguous and data.size==w*h*n and data.nbytes<=1<<20
        dt=types.MEMCPY_16BIT if bits==16 else types.MEMCPY_32BIT
        if read:runner.memcpy_d2h(data,ids[name],x,y,w,h,n,data_type=dt,**opts)
        else:runner.memcpy_h2d(ids[name],data,x,y,w,h,n,data_type=dt,**opts)
    def resident(name,data,x,y,w,h,n,bits,verify):
        if verify:
            actual=np.zeros_like(data);copy(name,actual,x,y,w,h,n,bits,True)
            if bits==16:actual&=65535
            if not np.array_equal(actual,data):raise ValueError(f'Resident {name} changed at {x},{y}')
        else:copy(name,data,x,y,w,h,n,bits)
    identity=np.zeros((1160,4,4),np.uint32);identity[:,:,1]=np.arange(1160,dtype=np.uint32)[:,None];identity[:,:,3]=65535
    identity[:1140,0,2]=np.arange(1140)//30;identity[:60,1,2]=38+np.arange(60)//30
    identity[64:1053,1,2]=np.arange(989)//43;identity[:8,3,2]=np.arange(8)
    nodes=sorted({*range(8),*(29+30*i for i in range(38)),*(106+43*i for i in range(23))})
    for i,y in enumerate(nodes):identity[y,2,2]=i
    copy('identity',identity,0,0,4,1160,4,32)
    def weights(verify=False):
        for kind,rows,contractions in [('qkv',40,30),('o',23,43)]:
            for row in range(rows):
                x,base=(row//38,30*(row%38)) if kind=='qkv' else (1,64+43*row)
                for offset in range(0,contractions,15):
                    h=min(15,contractions-offset)
                    data=fixture[kind+'_weights'][row,offset:offset+h].reshape(-1).astype(np.uint32)
                    resident('weights',data,x,base+offset,1,h,12288,16,verify)
                resident('bias',fixture[kind+'_bias'][row].astype(np.uint32),x,base+contractions-1,1,1,128,16,verify)
        resident('gain',fixture['gain'].astype(np.uint32),2,0,1,1,2880,16,verify)
        resident('frequencies',fixture['frequencies'].view(np.uint32),2,0,1,1,32,32,verify)
        resident('sinks',fixture['sinks'].reshape(-1).astype(np.uint32),3,0,1,8,8,16,verify)
    weights();log('all_original_weights_uploaded_once')
    records=[]
    for epoch,hidden in enumerate(fixture['hidden'],1):
        copy('hidden',hidden.view('<u4').copy(),0,1158,1,1,1440,32)
        commands=[(3,0),*((4,i) for i in range(40)),(5,0),*((6,i) for i in range(8)),(7,0),*((8,i) for i in range(23)),(9,0)]
        for phase,index in commands:runner.launch('step',np.uint16(phase),np.uint16(0),np.uint16(index),nonblock=False)
        log('fixed_attention_phases_dispatched',epoch=epoch,commands=len(commands))
        raw=np.zeros(1440,np.uint32);copy('hidden',raw,3,1159,1,1,1440,32,True)
        output=raw.view('<u2').copy();checks={'output':compare(output,fixture['expected_output'][epoch-1])}
        caches={}
        for name in ('keys','values'):
            data=np.zeros((8,96,64),np.uint32);copy(name,data,3,0,1,8,6144,16,True)
            actual=(data&65535).astype(np.uint16);expected=np.zeros((8,96,64),np.uint16)
            expected[:,:epoch]=fixture[name][:epoch].transpose(1,0,2)
            caches[name]=actual;checks[name]=compare(actual,expected)
        counters=np.zeros((1160,4,6),np.uint32);copy('counters',counters,0,0,4,1160,6,32,True)
        completed=bool(np.all(counters[:,:,0]==epoch) and np.all(counters[:,:,4]==epoch)
          and counters[0,2,1]==epoch*40 and counters[0,2,2]==epoch*23 and counters[0,2,3]==epoch*8
          and np.all(counters[:8,3,1]==epoch*8))
        record=dict(epoch=epoch,checks=checks,every_pe_completed=completed,persistent_kv_tokens=epoch)
        records.append(record);np.savez(f'actual-{epoch}.npz',output=output,counters=counters,**caches)
        Path('observations.json').write_text(json.dumps(records,indent=2)+'\n');log('epoch_compared',**record)
        if not(completed and all(v['passed'] for v in checks.values())):raise ValueError('Complete attention numerical/protocol acceptance failed')
    weights(True);resident('identity',identity,0,0,4,1160,4,32,True);log('all_original_weights_retained')
    return meta,records

def main():
    p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');a=p.parse_args()
    with runtime(a.physical) as (runner,types,order):meta,records=execute(runner,types,order)
    result=dict(passed=True,physical=a.physical,normal_stop=True,full_attention=True,full_model=False,
      scope=meta['scope'],west_to_east=True,weights_uploaded_once=True,all_weights_retained=True,
      host_intermediate_neural_computation=False,epochs=records)
    Path('result.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
