"""Complete resident 24-layer inference. Reference activations are never uploaded."""
import argparse,hashlib,json,time,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from backend import runtime
from checkpoint import Checkpoint
import resident_geometry as geo
from resident_loader import audit
from resident_identity import iter_identity
from resident_schedule import token_schedule

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def compare(actual,expected):
    a=(actual.astype(np.uint32)<<16).view(np.float32);b=(expected.astype(np.uint32)<<16).view(np.float32)
    def ordered(x):
        x=x.astype(np.int32);return np.where(x&32768,-(x&32767),x)
    ulp=np.abs(ordered(actual)-ordered(expected));delta=a.astype(np.float64)-b
    finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
    return dict(passed=bool(finite and (ulp<=1).all()),finite=finite,max_bf16_ulp=int(ulp.max()),
      bit_exact=int(np.count_nonzero(actual==expected)),values=int(actual.size),
      max_abs_error=float(np.max(np.abs(delta))),rms_error=float(np.sqrt(np.mean(delta**2))),
      relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(b.astype(np.float64)),1e-30)))

def execute(runner,types,order,model):
    meta=json.loads(Path('full-fixture.json').read_text());assert meta['initial_tokens']==[13225]
    assert digest(Path('full-fixture.npz'))==meta['fixture_sha256']
    with np.load('full-fixture.npz',allow_pickle=False) as f:fixture={k:f[k] for k in f.files}
    receipt=json.loads((model/'COMPLETE.json').read_text())
    assert receipt['revision']==meta['revision'] and receipt['tensors']==459 and receipt['payload_bytes']==13761264768
    for item in receipt['files']:
        assert item['publisher_verified'] and Path(item['file']).name==item['file']
        p=model/item['file'];assert p.stat().st_size==item['bytes'] and digest(p)==item['sha256']
    names=('identity','weights','scales','bias','gain','sinks','frequencies','token','hidden','output','ids','counters',
      'input','logits','probabilities','gate_debug','activation_debug')
    ids={n:runner.get_id(n) for n in names};opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
    started=time.monotonic()
    def log(phase,**data):print(json.dumps(dict(phase=phase,seconds=time.monotonic()-started,**data)),flush=True)
    def copy(name,data,x,y,w,h,n,bits,read=False):
        assert data.dtype==np.uint32 and data.flags.c_contiguous and data.size==w*h*n and data.nbytes<=1<<20
        dt=types.MEMCPY_16BIT if bits==16 else types.MEMCPY_32BIT
        if read:runner.memcpy_d2h(data,ids[name],x,y,w,h,n,data_type=dt,**opts)
        else:runner.memcpy_h2d(ids[name],data,x,y,w,h,n,data_type=dt,**opts)
    for y,h,data in iter_identity():copy('identity',data,0,y,geo.WIDTH,h,4,32)
    log('runtime_identities_initialized')
    transfers=0
    def upload(item):
        nonlocal transfers
        copy(item.symbol,item.data,item.x,item.y,item.width,item.height,item.count,16);transfers+=1
        if transfers%5000==0:log('original_weights_upload',transfers=transfers)
    loading=audit(Checkpoint(model),upload)
    Path('weight-load.json').write_text(json.dumps(loading,indent=2)+'\n')
    for layer in range(24):
        x,y=geo.controller(layer,'attention');copy('frequencies',fixture['frequencies'].view(np.uint32),x,y,1,1,32,32)
    log('all_459_original_tensors_uploaded_once',**{k:loading[k] for k in ('original_bytes','host_slot_bytes','transfers')})
    active=np.zeros((geo.HEIGHT,geo.WIDTH),bool);active[:,46:694]=True
    active[:1140,4:46]=True;active[:1140,694:736]=True
    active[0,736]=True;active[29:1140:30,736]=True;active[0,0]=True;active[0,749]=True
    tokens=[13225];records=[]
    for position,expected_token in enumerate(meta['expected_generated_tokens']):
        epoch=position+1;copy('token',np.array([tokens[-1]],np.uint32),0,0,1,1,1,32)
        tick=time.monotonic();commands=0
        for command in token_schedule(position):
            # A host submission trace is not a device-completion assertion.
            # It locates a blocked command without reading neural intermediates.
            if int(command.phase) in (1,3,10,12,14,15) or (int(command.phase)==16 and command.index==37):
                log('submitting_control',position=position,phase_id=int(command.phase),layer=command.layer,index=command.index,commands_submitted=commands)
            runner.launch('step',np.uint16(command.phase),np.uint16(command.layer),np.uint16(command.index),nonblock=False)
            commands+=1
        sampled=np.zeros(1,np.uint32);copy('token',sampled,749,0,1,1,1,32,True)
        seconds=time.monotonic()-tick;next_token=int(sampled[0]);tokens.append(next_token)
        log('full_token_received_at_east',position=position,token=next_token,seconds_for_commands_and_output=seconds,commands=commands)
        # Post-generation diagnostics only. These never provide a forward operand.
        actual={kind:np.zeros((24,2880),np.uint16) for kind in ('attention','output')}
        actual_ids=np.zeros((24,4),np.uint16);layers=[]
        diagnostic={kind:np.zeros((24,count),np.uint16) for kind,count in
          [('attention_norm',2880),('moe_norm',2880),('router_logits',32),('router_probabilities',4)]}
        diagnostic.update({kind:np.zeros((24,4,count),np.uint16) for kind,count in
          [('expert_gate',5760),('expert_activation',2880),('expert_output',2880)]})
        for layer in range(24):
            checks={}
            for kind,symbol,owner in [('attention','hidden','attention'),('output','output','join')]:
                x,y=geo.controller(layer,owner);raw=np.zeros(1440,np.uint32);copy(symbol,raw,x,y,1,1,1440,32,True)
                actual[kind][layer]=raw.view('<u2');checks[kind]=compare(actual[kind][layer],fixture[kind][position,layer])
            raw=np.zeros(4,np.uint32);x,y=geo.controller(layer,'moe');copy('ids',raw,x,y,1,1,4,16,True)
            actual_ids[layer]=(raw&65535).astype(np.uint16)
            b=geo.layer_x(layer)
            for kind,x,y,h,n in [('attention_norm',b,0,30,96),('moe_norm',b+3,1159,1,2880)]:
                raw=np.zeros(2880,np.uint32);copy('input',raw,x,y,1,h,n,32,True)
                if np.any(raw&65535):raise ValueError('Expanded normalized input is not exact BF16')
                diagnostic[kind][layer]=(raw>>16).astype(np.uint16)
            for kind,symbol,n in [('router_logits','logits',32),('router_probabilities','probabilities',4)]:
                raw=np.zeros(n,np.uint32);copy(symbol,raw,b+3,1159,1,1,n,16,True)
                diagnostic[kind][layer]=(raw&65535).astype(np.uint16)
            for rank,expert in enumerate(actual_ids[layer]):
                for kind,symbol,n in [('expert_gate','gate_debug',5760),('expert_activation','activation_debug',2880)]:
                    raw=np.zeros(n,np.uint32);copy(symbol,raw,b+14,36*int(expert),1,1,n,16,True)
                    diagnostic[kind][layer,rank]=(raw&65535).astype(np.uint16)
                raw=np.zeros(1440,np.uint32);copy('output',raw,b+25,36*int(expert),1,1,1440,32,True)
                diagnostic['expert_output'][layer,rank]=raw.view('<u2')
            layers.append(dict(layer=layer,checks=checks,selected_ids=actual_ids[layer].tolist(),
              original_ids=fixture['expected_ids'][position,layer].tolist(),ids_exact=bool(np.array_equal(actual_ids[layer],fixture['expected_ids'][position,layer]))))
        counters=np.zeros((geo.HEIGHT,geo.WIDTH,6),np.uint32)
        for y in range(0,geo.HEIGHT,32):
            h=min(32,geo.HEIGHT-y);copy('counters',counters[y:y+h],0,y,geo.WIDTH,h,6,32,True)
        expected=active.astype(np.uint32)*epoch
        completed=bool(np.array_equal(counters[:,:,0],expected) and np.array_equal(counters[:,:,4],expected))
        for layer in range(24):
            b=geo.layer_x(layer)
            completed=completed and bool(counters[0,b+2,1]==40*epoch and counters[0,b+2,2]==23*epoch
              and counters[0,b+2,3]==8*epoch and counters[0,b+26,5]==epoch
              and np.all(counters[:1152:36,b+25,5]==epoch) and np.all(counters[:8,b+3,1]==8*epoch))
        layer_match=all(row['ids_exact'] and all(c['passed'] for c in row['checks'].values()) for row in layers)
        record=dict(position=position,input_token=tokens[-2],generated_token=next_token,expected_token=expected_token,
          token_exact=next_token==expected_token,every_active_pe_completed=completed,layers=layers,
          forward_command_count=commands,forward_host_seconds=seconds,persistent_kv_tokens=epoch,
          strict_all_layer_checks_passed=layer_match)
        records.append(record);np.savez(f'actual-{epoch}.npz',**actual,ids=actual_ids,counters=counters,**diagnostic)
        Path('observations.json').write_text(json.dumps(records,indent=2)+'\n')
        log('full_token_compared',position=position,token_exact=record['token_exact'],every_active_pe_completed=completed,all_layer_checks_passed=layer_match)
        if not completed:raise ValueError('Full-model protocol acceptance failed; diagnostics preserved')
        # Keep strict comparisons intact, but collect both real autoregressive
        # steps and retention before offline, same-input operator qualification.
        # No reference/diagnostic tensor is ever forwarded back into the wafer.
    verified=0
    def retain(item):
        nonlocal verified
        data=np.zeros_like(item.data);copy(item.symbol,data,item.x,item.y,item.width,item.height,item.count,16,True)
        if not np.array_equal(data&65535,item.data):raise ValueError(f'Resident weight changed: {item.tensor} at {item.x},{item.y}')
        verified+=1
        if verified%5000==0:log('resident_weights_readback',transfers=verified)
    retention=audit(Checkpoint(model),retain)
    for y,h,data in iter_identity():
        actual=np.zeros_like(data);copy('identity',actual,0,y,geo.WIDTH,h,4,32,True)
        if not np.array_equal(actual,data):raise ValueError('Runtime identities changed')
    for layer in range(24):
        actual=np.zeros(32,np.uint32);x,y=geo.controller(layer,'attention');copy('frequencies',actual,x,y,1,1,32,32,True)
        if not np.array_equal(actual,fixture['frequencies'].view(np.uint32)):raise ValueError('RoPE frequencies changed')
    Path('weight-retention.json').write_text(json.dumps(retention,indent=2)+'\n');log('all_original_weights_verified_unchanged')
    strict=all(r['strict_all_layer_checks_passed'] and r['token_exact'] for r in records)
    return dict(passed=strict,strict_original_reference_passed=strict,
      physical_capture_complete=True,numerical_qualification_pending=not strict,
      full_model=True,model=meta['model'],revision=meta['revision'],tokens=tokens,
      scope='complete original 24-layer, 32-expert GPT-OSS-20B inference on one WSE-3; 96-token runtime KV capacity',
      weights_uploaded_once=True,all_weights_retained=True,west_to_east=True,host_intermediate_neural_computation=False,
      diagnostic_layer_reads='after each complete token only; never forwarded back into computation',records=records)

def main():
    p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true')
    p.add_argument('--model',type=Path,default=Path('/srv/gpt-oss20b-hardware/model'));a=p.parse_args()
    with runtime(a.physical) as (runner,types,order):result=execute(runner,types,order,a.model)
    result.update(physical=a.physical,normal_stop=True)
    Path('result.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
