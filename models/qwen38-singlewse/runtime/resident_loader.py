"""Initialize all original text weights once, with bounded copies and readback.

No method here is used between prompt/decode positions. Host arithmetic is
limited to packing checkpoint bytes and copying original actor parameters.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
from weights import OriginalWeights
from resident_plan import load

def upload(runner,dtype,order,root,model):
    root=Path(root);plan=load(root);hub=json.loads((root/'hub.json').read_text())
    pins={s['rfilename']:s['lfs']['sha256'] for s in hub['siblings'] if s['rfilename'].endswith('.safetensors')}
    weights=OriginalWeights(model,plan['tensors'],pins);started=time.monotonic();copies=0;uploaded=0
    names=['config','weights','scales','data','conv','gate','gain','freq','program','matrices','io','prompt','controls','generated','result','trace']
    ids={n:runner.get_id(n) for n in names};opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
    def put(name,value,x,y,width=1,height=1,bits=32):
        nonlocal copies,uploaded
        value=np.ascontiguousarray(value)
        if value.dtype==np.uint16:value=value.astype(np.uint32)
        if value.nbytes>16<<20:raise ValueError('Host transport buffer ceiling')
        assert value.size%(width*height)==0
        runner.memcpy_h2d(ids[name],value.reshape(-1),x,y,width,height,value.size//(width*height),
            data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
        copies+=1;uploaded+=value.nbytes
    def get(name,count,x,y,bits=32):
        value=np.zeros(count,np.uint32)
        runner.memcpy_d2h(value,ids[name],x,y,1,1,count,
            data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
        return value&65535 if bits==16 else value
    def progress(stage,**extra):
        row=dict(stage=stage,copies=copies,host_word_bytes=uploaded,seconds=time.monotonic()-started,**extra)
        (root/'initialization-progress.json').write_text(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
    # Configurations and tables carry no neural operands.
    for y in range(0,1160,4):put('config',plan['config'][y:y+4],0,y,750,4)
    for xy in [[0,0],[748,89],[744,154],[720,709],[749,1159]]:
        x,y=xy;assert np.array_equal(get('config',8,x,y),plan['config'][y,x])
    program=json.loads((root/'forward-program.json').read_text())
    assert program['role_plan_sha256']==hashlib.sha256((root/'role-plan-96.json').read_bytes()).hexdigest()
    assert program['matrix_table_sha256']==hashlib.sha256((root/'device-matrices-bf16-140.json').read_bytes()).hexdigest()
    for name,array in [('program',program['instructions']),('matrices',plan['matrices']['table']),('io',[v['coordinate'] for v in program['io']])]:
        data=np.array(array,np.uint16).reshape(-1);put(name,data,749,1159,bits=16)
        assert np.array_equal(get(name,data.size,749,1159,16),data)
    progress('configuration_verified',role_sha256=plan['role_sha256'],config_sha256=plan['config_sha256'])
    tensor_count=0;tile_count=0;retention_samples=0;loaded_names=set()
    with (root/'weight-initialization.jsonl').open('x') as receipt:
        for name,a in plan['atlas']['assignments'].items():
            if 'shared_with' in a:
                # Original FP8 scales are uploaded with their parent strip.
                assert name==a['shared_with']+'_scale_inv'
                assert plan['atlas']['assignments'][a['shared_with']]['role']=='fp8_matrix'
                continue
            if 'strips' in a:
                m,k=a['shape'];n=k//128
                for ordinal,x,y in a['strips']:
                    packed=weights.strip(name,ordinal,bf16_rows=140)
                    assert packed['weights'].shape[0]==n and packed['valid_rows']==int(plan['config'][y,x,4])
                    put('weights',packed['weights'],x,y,n,1,16)
                    if packed['scales'] is not None:put('scales',packed['scales'],x,y,n,1)
                    # Deterministic first/last strip endpoint samples, plus
                    # complete publisher/source packing checks, bound I/O cost.
                    if ordinal in (0,len(a['strips'])-1):
                        for endpoint in (0,n-1):
                            observed=get('weights',packed['weights'].shape[1],x+endpoint,y,16)
                            assert np.array_equal(observed,packed['weights'][endpoint]);retention_samples+=1
                            if packed['scales'] is not None:
                                assert np.array_equal(get('scales',3,x+endpoint,y),packed['scales'][endpoint].view(np.uint32))
                    tile_count+=n
                tensor_count+=1+(a['role']=='fp8_matrix')
                loaded_names.add(name)
                if a['role']=='fp8_matrix':loaded_names.add(name+'_scale_inv')
            else:
                bits=weights.small_tensor(name).reshape(-1)
                for index,(x,y) in enumerate(a['coordinates']):
                    part=bits[index*16384:(index+1)*16384];put('weights',part,x,y,bits=16)
                    assert np.array_equal(get('weights',part.size,x,y,16),part);retention_samples+=1;tile_count+=1
                tensor_count+=1
                loaded_names.add(name)
            receipt.write(json.dumps(dict(tensor=name,original_bytes=plan['tensors'][name]['bytes'],publisher_shard=plan['tensors'][name]['shard'],loaded=True))+'\n');receipt.flush()
            progress('original_weights',tensor=name,text_tensors_loaded=tensor_count,weight_pes_loaded=tile_count)
    assert tensor_count==1251 and tile_count==849313
    assert loaded_names==set(plan['atlas']['assignments'])
    # Original duplicated parameters colocate with their dependent state groups.
    for layer in range(64):
        if layer%4==3:continue
        prefix=f'model.language_model.layers.{layer}.linear_attn.'
        conv=weights.small_tensor(prefix+'conv1d.weight').reshape(10240,4)
        alog=weights.small_tensor(prefix+'A_log');bias=weights.small_tensor(prefix+'dt_bias');gain=weights.small_tensor(prefix+'norm.weight')
        for group in (g for g in plan['roles']['gdn'] if g['layer']==layer):
            x,y=group['controller'];head=group['value_head'];starts=group['conv_channel_starts']
            copied=np.concatenate([conv[s:s+128] for s in starts]).reshape(-1)
            gate=np.array([alog[head],bias[head]],np.uint16)
            for name,data in [('conv',copied),('gate',gate),('gain',gain)]:
                put(name,data,x,y,bits=16);assert np.array_equal(get(name,data.size,x,y,16),data)
        progress('gdn_parameter_copies',layer=layer)
    frequency=json.loads((root/'rotary-frequencies.json').read_text())
    assert frequency['source']=='attention-group-hw-001' and len(frequency['u32'])==32
    freq=np.array(frequency['u32'],np.uint32).view(np.float32)
    assert np.isfinite(freq).all() and (freq>0).all()
    for group in plan['roles']['attention']:
        prefix=f"model.language_model.layers.{group['layer']}.self_attn."
        gain=np.concatenate([weights.small_tensor(prefix+'q_norm.weight'),weights.small_tensor(prefix+'k_norm.weight')])
        x,y=group['controller'];put('gain',gain,x,y,bits=16);put('freq',freq,x,y)
        assert np.array_equal(get('gain',512,x,y,16),gain)
        assert np.array_equal(get('freq',32,x,y),freq.view(np.uint32))
    for shard,stamp in weights.verified.items():
        assert weights.stamp((Path(model)/shard).stat())==stamp
    complete=dict(passed=True,phase='initialization_only',complete_physical_model=False,text_tensors_loaded=tensor_count,
        original_weight_pes_loaded=tile_count,original_weight_retention_samples=retention_samples,
        original_weight_readback_scope='Full other tensors, first/last strip endpoints of every matrix; not every matrix PE read back',
        actor_parameter_copy_readback='complete',role_sha256=plan['role_sha256'],config_sha256=plan['config_sha256'],
        host_word_bytes=uploaded,copies=copies,seconds=time.monotonic()-started,per_token_weight_upload=False)
    (root/'initialization.json').write_text(json.dumps(complete,indent=2)+'\n')
    return ids,complete
