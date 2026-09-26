"""Whole original decoder layer; attention feeds MoE entirely on the wafer."""
import argparse,hashlib,json,time,sys
from pathlib import Path
from collections import Counter
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from backend import runtime
from checkpoint import Checkpoint
import resident_geometry as geo
from resident_loader import dense_weights,dense_bias,vector,transfer,experts

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def layer_weights(c):
    prefix='model.layers.0.'
    for projection,offset in [('q_proj',0),('k_proj',32),('v_proj',36)]:
        coordinate=lambda r,k,o=offset:geo.qkv_tile(0,r+o,k)
        yield from dense_weights(c,prefix+'self_attn.'+projection+'.weight',coordinate)
        yield from dense_bias(c,prefix+'self_attn.'+projection+'.bias',coordinate,30)
    coordinate=lambda r,k:geo.o_tile(0,r,k)
    yield from dense_weights(c,prefix+'self_attn.o_proj.weight',coordinate)
    yield from dense_bias(c,prefix+'self_attn.o_proj.bias',coordinate,43)
    coordinate=lambda r,k:geo.router_tile(0,k)
    yield from dense_weights(c,prefix+'mlp.router.weight',coordinate,row_block=32)
    yield from dense_bias(c,prefix+'mlp.router.bias',coordinate,30,row_block=32)
    yield from vector(c,prefix+'input_layernorm.weight','gain',geo.controller(0,'attention'))
    yield from vector(c,prefix+'post_attention_layernorm.weight','gain',geo.controller(0,'moe'))
    name=prefix+'self_attn.sinks';sinks=c.tensor(name)
    for head in range(8):yield transfer('sinks',(49,head),1,1,8,sinks[head*8:(head+1)*8],name,16)
    yield from experts(c,0)

def compare(actual,expected):
    a=(actual.astype(np.uint32)<<16).view(np.float32);b=(expected.astype(np.uint32)<<16).view(np.float32)
    def ordered(x):
        x=x.astype(np.int32);return np.where(x&32768,-(x&32767),x)
    ulp=np.abs(ordered(actual)-ordered(expected))
    return dict(passed=bool(np.isfinite(a).all() and np.isfinite(b).all() and (ulp<=1).all()),
      max_bf16_ulp=int(ulp.max()),bit_exact=int(np.count_nonzero(actual==expected)),values=int(actual.size),max_abs_error=float(np.max(np.abs(a-b))))

def execute(runner,types,order,model):
    meta=json.loads(Path('full-fixture.json').read_text());assert digest(Path('full-fixture.npz'))==meta['fixture_sha256']
    with np.load('full-fixture.npz',allow_pickle=False) as f:fixture={k:f[k] for k in f.files}
    receipt=json.loads((model/'COMPLETE.json').read_text());assert receipt['revision']==meta['revision'] and receipt['tensors']==459
    c=Checkpoint(model);vectors=np.stack([np.array(c.row_slice('model.embed_tokens.weight',token)[0],copy=True) for token in (13225,11)])
    ids={n:runner.get_id(n) for n in ('identity','weights','scales','bias','gain','sinks','frequencies','hidden','output','ids','counters')}
    opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False);start=time.monotonic()
    def log(phase,**data):print(json.dumps(dict(phase=phase,seconds=time.monotonic()-start,**data)),flush=True)
    def copy(name,data,x,y,w,h,n,bits,read=False):
        assert data.dtype==np.uint32 and data.flags.c_contiguous and data.size==w*h*n and data.nbytes<=1<<20
        dt=types.MEMCPY_16BIT if bits==16 else types.MEMCPY_32BIT
        if read:runner.memcpy_d2h(data,ids[name],x,y,w,h,n,data_type=dt,**opts)
        else:runner.memcpy_h2d(ids[name],data,x,y,w,h,n,data_type=dt,**opts)
    identity=np.zeros((1160,27,4),np.uint32)
    for y in range(1160):
        for x in range(27):
            role=geo.role_at(46+x,y);identity[y,x]=[0,y,role.row if role.row>=0 else 65535,role.expert if role.expert>=0 else 65535]
    copy('identity',identity,0,0,27,1160,4,32)
    def weights(verify=False):
        coverage=Counter();count=0
        for item in layer_weights(c):
            coverage[item.tensor]+=item.source_bytes;count+=1
            if verify:
                actual=np.zeros_like(item.data);copy(item.symbol,actual,item.x-46,item.y,item.width,item.height,item.count,16,True)
                if not np.array_equal(actual&65535,item.data):raise ValueError('Resident layer tensor changed: '+item.tensor)
            else:copy(item.symbol,item.data,item.x-46,item.y,item.width,item.height,item.count,16)
        expected={name:spec['data_offsets'][1]-spec['data_offsets'][0] for _,header in c.headers.values() for name,spec in header.items() if name.startswith('model.layers.0.')}
        assert len(expected)==19 and dict(coverage)==expected
        log('all_layer_weights_retained' if verify else 'all_layer_weights_uploaded_once',transfers=count,original_bytes=sum(coverage.values()),tensors=len(coverage))
    weights();copy('frequencies',fixture['frequencies'].view(np.uint32),2,0,1,1,32,32)
    records=[]
    for epoch,hidden in enumerate(vectors,1):
        copy('hidden',hidden.view('<u4').copy(),0,1158,1,1,1440,32)
        commands=[(3,0),*((4,i) for i in range(40)),(5,0),*((6,i) for i in range(8)),(7,0),*((8,i) for i in range(23)),*((phase,0) for phase in range(9,14))]
        for phase,index in commands:runner.launch('step',np.uint16(phase),np.uint16(0),np.uint16(index),nonblock=False)
        values={};checks={}
        for kind,symbol,x in [('attention','hidden',2),('output','output',26)]:
            data=np.zeros(1440,np.uint32);copy(symbol,data,x,0,1,1,1440,32,True)
            values[kind]=data.view('<u2').copy();checks[kind]=compare(values[kind],fixture[kind][epoch-1,0])
        selected=np.zeros(4,np.uint32);copy('ids',selected,3,1159,1,1,4,16,True);selected&=65535
        ids_exact=bool(np.array_equal(selected,fixture['expected_ids'][epoch-1,0]))
        counters=np.zeros((1160,27,6),np.uint32);copy('counters',counters,0,0,27,1160,6,32,True)
        completed=bool(np.all(counters[:,:,0]==epoch) and np.all(counters[:,:,4]==epoch)
          and np.all(counters[:1152:36,25,5]==epoch) and np.all(counters[:8,3,1]==8*epoch)
          and counters[0,2,1]==40*epoch and counters[0,2,2]==23*epoch and counters[0,2,3]==8*epoch)
        record=dict(epoch=epoch,checks=checks,selected_ids=selected.tolist(),ids_exact=ids_exact,every_pe_completed=completed,persistent_kv_tokens=epoch)
        records.append(record);np.savez(f'actual-{epoch}.npz',**values,ids=selected,counters=counters)
        Path('observations.json').write_text(json.dumps(records,indent=2)+'\n');log('complete_decoder_compared',**record)
        if not(completed and ids_exact and all(v['passed'] for v in checks.values())):raise ValueError('Whole decoder layer acceptance failed')
    weights(True)
    actual=np.zeros_like(identity);copy('identity',actual,0,0,27,1160,4,32,True);assert np.array_equal(actual,identity)
    return dict(passed=True,full_model=False,full_decoder_layer=True,full_attention=True,full_moe_layer=True,
      scope='complete original layer-0 decoder: attention directly feeds 32-resident-expert MoE; two tokens with persistent KV',
      west_to_east=True,weights_uploaded_once=True,all_weights_retained=True,host_intermediate_neural_computation=False,epochs=records)

def main():
    p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');a=p.parse_args()
    with runtime(a.physical) as (runner,types,order):result=execute(runner,types,order,Path('/srv/gpt-oss20b-hardware/model'))
    result.update(physical=a.physical,normal_stop=True);Path('result.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
