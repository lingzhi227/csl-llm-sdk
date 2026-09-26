"""Whole MoE layer, original 32 resident experts; four control-only launches/input."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from backend import runtime

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(4<<20),b''):h.update(data)
    return h.hexdigest()

def compare(actual,expected):
    a=(actual.astype(np.uint32)<<16).view(np.float32)
    b=(expected.astype(np.uint32)<<16).view(np.float32)
    def ordered(x):
        x=x.astype(np.int32);return np.where(x&32768,-(x&32767),x)
    ulp=np.abs(ordered(actual)-ordered(expected))
    return dict(passed=bool(np.isfinite(a).all() and np.isfinite(b).all() and (ulp<=1).all()),
      max_bf16_ulp=int(ulp.max()),bit_exact=int(np.count_nonzero(actual==expected)),
      values=int(actual.size),max_abs_error=float(np.max(np.abs(a-b))))

def execute(runner,types,order):
    meta=json.loads(Path('moe-fixture.json').read_text())
    assert meta['experts']==32 and meta['layer']==0 and meta['application_pes']==31320
    assert digest(Path('moe-fixture.npz'))==meta['fixture_sha256']
    with np.load('moe-fixture.npz',allow_pickle=False) as f:fixture={k:f[k] for k in f.files}
    symbols=('identity','weights','scales','bias','gain','hidden','input','output','counters','logits','ids','probabilities','routing')
    ids={n:runner.get_id(n) for n in symbols}
    opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
    start=time.monotonic()
    def log(phase,**data):print(json.dumps(dict(phase=phase,seconds=time.monotonic()-start,**data)),flush=True)
    def copy(name,data,x,y,w,h,n,bits,read=False):
        assert data.dtype==np.uint32 and data.flags.c_contiguous and data.size==w*h*n and data.nbytes<=1<<20
        dt=types.MEMCPY_16BIT if bits==16 else types.MEMCPY_32BIT
        if read:runner.memcpy_d2h(data,ids[name],x,y,w,h,n,data_type=dt,**opts)
        else:runner.memcpy_h2d(ids[name],data,x,y,w,h,n,data_type=dt,**opts)
    def resident(name,slots,x,y,w,h,n,bits,verify):
        if verify:
            actual=np.zeros_like(slots);copy(name,actual,x,y,w,h,n,bits,True)
            if bits==16:actual&=65535
            if not np.array_equal(actual,slots):raise ValueError(f'Resident {name} changed at {x},{y}')
        else:copy(name,slots,x,y,w,h,n,bits)
    identity=np.zeros((1160,27,4),np.uint32)
    identity[:,:,1]=np.arange(1160,dtype=np.uint32)[:,None]
    identity[:,:,2]=(np.arange(1160,dtype=np.uint32)%36)[:,None]
    identity[:,:,3]=65535;identity[:1152,4:,3]=(np.arange(1152,dtype=np.uint32)//36)[:,None]
    copy('identity',identity,0,0,27,1160,4,32)
    def weights(verify=False):
        resident('weights',fixture['router_weights'].reshape(-1).astype(np.uint32),1,1054,1,30,3072,16,verify)
        resident('bias',fixture['router_bias'].astype(np.uint32),1,1083,1,1,32,16,verify)
        resident('gain',fixture['gain'].astype(np.uint32),3,1159,1,1,2880,16,verify)
        for expert in range(32):
            for kind,rows,x in [('gate',36,4),('down',18,15)]:
                for offset in range(0,rows,2):
                    for field,symbol,count in [('blocks','weights',11520),('scales','scales',720)]:
                        raw=fixture[kind+'_'+field][expert,offset:offset+2]
                        slots=raw.reshape(-1).view('<u2').astype(np.uint32)
                        resident(symbol,slots,x,36*expert+offset,10,2,count,16,verify)
                    slots=fixture[kind+'_bias'][expert,offset:offset+2].reshape(-1).astype(np.uint32)
                    resident('bias',slots,x+9,36*expert+offset,1,2,160,16,verify)
            if (expert+1)%8==0:log('weights_verified' if verify else 'weights_uploaded',experts=expert+1)
    weights();log('all_original_weights_uploaded_once')
    records=[]
    for epoch,hidden in enumerate(fixture['hidden'],1):
        copy('hidden',hidden.view('<u4').copy(),0,1158,1,1,1440,32)
        for phase in (10,11,12,13):
            runner.launch('step',np.uint16(phase),np.uint16(0),np.uint16(0),nonblock=False)
        log('four_control_phases_dispatched',epoch=epoch)
        packed=np.zeros(1440,np.uint32);copy('output',packed,26,0,1,1,1440,32,True)
        output=packed.view('<u2').copy();checks={'output':compare(output,fixture['expected_output'][epoch-1])}
        norm=np.zeros(2880,np.uint32);copy('input',norm,3,1159,1,1,2880,32,True)
        checks['normalized_exact']=bool(np.array_equal(norm,fixture['normalized'][epoch-1].astype(np.uint32)<<16))
        values={}
        for name,count in [('logits',32),('ids',4),('probabilities',4)]:
            raw=np.zeros(count,np.uint32);copy(name,raw,3,1159,1,1,count,16,True)
            values[name]=(raw&65535).astype(np.uint16)
            checks[name+'_exact']=bool(np.array_equal(values[name],fixture[name][epoch-1]))
        counters=np.zeros((1160,27,6),np.uint32);copy('counters',counters,0,0,27,1160,6,32,True)
        completed=bool(np.all(counters[:,:,0]==epoch) and np.all(counters[:,:,4]==epoch)
                       and np.all(counters[:1152:36,25,5]==epoch) and counters[1159,3,5]==epoch)
        routes=np.zeros((32,8),np.uint32)
        for expert in range(32):copy('routing',routes[expert],26,36*expert,1,1,8,32,True)
        expected_route=np.zeros(8,np.uint32);expected_route[0]=epoch
        expected_route[1:5]=fixture['ids'][epoch-1];expected_route[5:7]=fixture['probabilities'][epoch-1].view('<u4')
        expected_route[7]=sum(1<<int(i) for i in fixture['ids'][epoch-1])
        metadata=bool(np.all(routes==expected_route))
        record=dict(epoch=epoch,selected_ids=values['ids'].tolist(),checks=checks,
          every_pe_completed=completed,all_expert_metadata_exact=metadata)
        records.append(record);np.savez(f'actual-{epoch}.npz',output=output,normalized=norm,counters=counters,routing=routes,**values)
        Path('observations.json').write_text(json.dumps(records,indent=2)+'\n');log('epoch_compared',**record)
        if not(completed and metadata and checks['output']['passed'] and all(v for k,v in checks.items() if k!='output')):
            raise ValueError('Whole MoE layer numerical/protocol acceptance failed')
    weights(True);resident('identity',identity,0,0,27,1160,4,32,True)
    log('all_original_weights_and_identities_retained')
    return meta,records

def main():
    p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');a=p.parse_args()
    with runtime(a.physical) as (runner,types,order):meta,records=execute(runner,types,order)
    result=dict(passed=True,physical=a.physical,normal_stop=True,full_model=False,full_moe_layer=True,
      scope=meta['scope'],west_to_east=True,weights_uploaded_once=True,all_weights_retained=True,
      host_intermediate_neural_computation=False,epochs=records)
    Path('result.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
