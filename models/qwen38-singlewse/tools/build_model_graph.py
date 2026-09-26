"""Bind every original text tensor to the complete device graph under development.

This is validated lowering input, not an executable or accepted CSL model.
BF16 boundaries follow the pinned eager Transformers arithmetic bodies.
"""
import hashlib,json,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def build():
    specs=json.loads((ROOT/'configs/tensors.json').read_text())['tensors']
    text=set(json.loads((ROOT/'configs/capacity.json').read_text())['assignments'])
    nodes=[];values={'token':dict(shape=[1],dtype='U32')};used=set()
    def op(kind,inputs,outputs,weights=(),**attrs):
        assert all(v in values for v in inputs)
        assert all(w in text for w in weights)
        ids=[]
        for shape,dtype in outputs:
            name='v'+str(len(values));values[name]=dict(shape=shape,dtype=dtype)
            ids.append(name)
        used.update(weights)
        nodes.append(dict(id=len(nodes),op=kind,inputs=inputs,outputs=ids,weights=list(weights),attributes=attrs))
        return ids[0] if len(ids)==1 else ids
    def vector(n,dtype='BF16'):return [([n],dtype)]
    def norm(v,w,n):
        assert specs[w]['shape']==[n]
        return op('zero_centered_rmsnorm',[v],vector(n),[w],epsilon=1e-6,
                  arithmetic='FP32 norm and multiply by (1+FP32 weight), then BF16')
    def projection(v,prefix):
        w=prefix+'.weight';rows,cols=specs[w]['shape']
        assert values[v]['shape']==[cols]
        if specs[w]['dtype']=='F8_E4M3':
            scale=w+'_scale_inv';assert specs[scale]['shape']==[math.ceil(rows/128),math.ceil(cols/128)]
            return op('fp8_projection',[v],vector(rows),[w,scale],activation_group=128,
                weight_block=[128,128],activation='E4M3FN nearest even; max(absmax,1e-10)/448',
                accumulation='FP32 ascending128-wide contraction blocks; BF16 only after complete projection')
        assert specs[w]['dtype']=='BF16'
        return op('bf16_projection',[v],vector(rows),[w],accumulation='FP32, then BF16; no FP8 input quantization')
    h=op('embedding_lookup',['token'],vector(5120),['model.language_model.embed_tokens.weight'],vocabulary=248320)
    for layer in range(64):
        p=f'model.language_model.layers.{layer}';residual=h
        n=norm(h,p+'.input_layernorm.weight',5120)
        if layer%4!=3:
            a=p+'.linear_attn'
            qkv=projection(n,a+'.in_proj_qkv');z=projection(n,a+'.in_proj_z')
            av=projection(n,a+'.in_proj_a');bv=projection(n,a+'.in_proj_b')
            conv=op('causal_depthwise_conv_silu',[qkv],vector(10240),[a+'.conv1d.weight'],
                state=f'conv.{layer}',kernel=4,channels=10240,
                arithmetic='BF16 projected history; FP32 convolution sum to BF16; SiLU to BF16')
            q,k,v=op('delta_split_normalize_repeat',[conv],vector(6144,'FP32')+vector(6144,'FP32')+vector(6144),
                q_channels=[0,2048],k_channels=[2048,4096],v_channels=[4096,10240],
                key_heads=16,value_heads=48,head_dimension=128,repeat_interleave=3,epsilon=1e-6,
                arithmetic='FP32 Q/K L2 normalization; Q then divided by sqrt128')
            decay,beta=op('delta_gates',[av,bv],vector(48,'FP32')+vector(48),[a+'.A_log',a+'.dt_bias'],
                arithmetic='g=-exp(FP32 A_log)*softplus(FP32 a+dt_bias); FP32 exp(g); BF16 sigmoid(b)')
            core=op('gated_delta_recurrence',[q,k,v,decay,beta],vector(6144),state=f'recurrent.{layer}',
                heads=48,state_shape=[48,128,128],state_dtype='FP32',partition='two full-key,value64 shards per head',
                arithmetic='decay state, predict K^T state, beta correction, rank1 update, Q^T state; BF16 output')
            gated=op('gated_rmsnorm',[core,z],vector(6144),[a+'.norm.weight'],heads=48,head_dimension=128,epsilon=1e-6,
                arithmetic='FP32 norm -> BF16; multiply BF16 gain -> BF16; multiply FP32 SiLU(z) -> BF16')
            attn=projection(gated,a+'.out_proj')
        else:
            a=p+'.self_attn';qg=projection(n,a+'.q_proj');k=projection(n,a+'.k_proj');v=projection(n,a+'.v_proj')
            q,gate=op('split_attention_query_gate',[qg],vector(6144)+vector(6144),
                reshape=[24,512],split_last_axis=[256,256],note='Per-head split, not two contiguous6144-vector halves')
            q=op('per_head_zero_centered_rmsnorm',[q],vector(6144),[a+'.q_norm.weight'],heads=24,dimension=256,epsilon=1e-6)
            k=op('per_head_zero_centered_rmsnorm',[k],vector(1024),[a+'.k_norm.weight'],heads=4,dimension=256,epsilon=1e-6)
            q,k=op('partial_rotary_position',[q,k],vector(6144)+vector(1024),dimension=64,head_dimension=256,
                theta=10000000,position='request token position',arithmetic='Original BF16 cos/sin and BF16 multiply/add boundaries')
            op('kv_append',[k,v],[],state=f'kv.{layer}',context=96,heads=4,dimension=256,dtype='BF16')
            context=op('full_causal_attention',[q],vector(6144),state=f'kv.{layer}',query_heads=24,kv_heads=4,
                dimension=256,scale=0.0625,causal=True,
                arithmetic='FP32 dot -> BF16; scaling to BF16; FP32 softmax -> BF16; value dot -> BF16')
            gated=op('attention_sigmoid_gate',[context,gate],vector(6144),
                arithmetic='BF16 sigmoid(gate), BF16 product; pinned official eager implementation')
            attn=projection(gated,a+'.o_proj')
        h=op('residual_add',[residual,attn],vector(5120),arithmetic='BF16 output')
        residual=h;n=norm(h,p+'.post_attention_layernorm.weight',5120)
        g=projection(n,p+'.mlp.gate_proj');u=projection(n,p+'.mlp.up_proj')
        product=op('silu_multiply',[g,u],vector(17408),arithmetic='SiLU(gate) to BF16; multiply up to BF16')
        down=projection(product,p+'.mlp.down_proj')
        h=op('residual_add',[residual,down],vector(5120),arithmetic='BF16 output')
    h=norm(h,'model.language_model.norm.weight',5120)
    logits=projection(h,'lm_head')
    selected=op('full_vocabulary_argmax',[logits],vector(1,'U32'),vocabulary=248320,tie='Lowest token ID',device_feedback='embedding_lookup')
    assert used==text,(sorted(text-used),sorted(used-text))
    assert len(text)==1251
    assert sum(n['op']=='fp8_projection' for n in nodes)==400
    assert sum(n['op']=='gated_delta_recurrence' for n in nodes)==48
    assert sum(n['op']=='full_causal_attention' for n in nodes)==16
    # Last-use analysis is independent of a bank allocator and excludes weights/state.
    last_use={v:-1 for v in values}
    for node in nodes:
        for name in node['inputs']:last_use[name]=node['id']
    last_use[selected]=len(nodes)
    live={'token'};peak=4
    for node in nodes:
        live.update(node['outputs'])
        size=sum(math.prod(values[v]['shape'])*(2 if values[v]['dtype']=='BF16' else 4) for v in live)
        peak=max(peak,size)
        live={v for v in live if last_use[v]>node['id']}
    return dict(status='Validated complete tensor/semantic graph; CSL lowering and execution incomplete',full_model_complete=False,
        model='Qwen/Qwen3.8-27B-FP8',revision='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a',
        context=96,weights_bound=len(used),operations=len(nodes),transient_value_peak_bytes=peak,
        source_sha256=hashlib.sha256((ROOT/'reference/sources/modeling_qwen3_5.py').read_bytes()).hexdigest(),
        prefill='Device sequential recurrence requires comparison with pinned chunk-prefill reference before acceptance',
        nodes=nodes,values=values,selected_token=selected)

if __name__=='__main__':
    result=build();(ROOT/'configs/model-graph.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['nodes','values']},indent=2))
