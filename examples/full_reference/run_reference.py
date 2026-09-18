"""Unexecuted bounded reference: original full text layers, layerwise weight reads.

The entry/guard must qualify the CPU environment and checkpoint before this runs.
It is a reference only, never a substitute for device neural computation.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

ROOT=Path(__file__).resolve().parent


class Evidence:
    def __init__(self,root,check):
        self.root=root;root.mkdir(exist_ok=False);self.check=check;self.files={};self.ranges={};self.context=None
    def tensors(self,name,values):
        import numpy as np
        import torch
        self.check();arrays={};types={}
        for key,value in values.items():
            if value is None:continue
            value=value.detach().contiguous().cpu()
            if value.dtype==torch.bfloat16:array=value.view(torch.int16).numpy().view('<u2')
            elif value.dtype==torch.float32:array=value.numpy()
            else:raise ValueError('Unexpected reference state dtype')
            arrays[key]=array;types[key]=str(value.dtype)
        if sum(a.nbytes for a in arrays.values())>16<<20:raise ValueError('Reference evidence file bound')
        p=self.root/(name+'.npz')
        with p.open('xb') as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
        self.files[p.name]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),dtypes=types)
    def observe(self,name,value):
        import torch
        if not isinstance(value,torch.Tensor):return
        x=value.detach().float()
        if not torch.isfinite(x).all():raise ValueError('Nonfinite official observation: '+name)
        record=dict(dtype=str(value.dtype),shape=list(value.shape),min=float(x.min()),max=float(x.max()),max_abs=float(x.abs().max()))
        self.ranges[name]=record
    def state(self,step,stage,cache,hidden):
        lo,hi=((0,20),(20,44),(44,64))[stage]
        self.tensors(f'step{step}-stage{stage}-hidden',dict(hidden=hidden))
        for i in range(lo,hi):
            layer=cache.layers[i]
            if i%4==3:
                self.tensors(f'step{step}-layer{i}-state',dict(keys=layer.keys,values=layer.values))
            else:
                self.tensors(f'step{step}-layer{i}-state',dict(conv=layer.conv_states[0],recurrent=layer.recurrent_states[0]))


def run(check):
    import platform
    import numpy as np
    import torch
    from torch import nn
    from tokenizers import Tokenizer
    from checkpoint import Checkpoint
    from pinned import load,original_cache
    from restore import restore
    if platform.python_version()!='3.11.14' or torch.__version__!='2.4.0+cu121' or np.__version__!='1.25.0':
        raise ValueError('Accepted reference CPU environment changed')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    plan=json.loads((ROOT/'checkpoint-plan.json').read_bytes());cold=Path(plan['destination'])
    headers=json.loads((ROOT/'metadata/tensor-headers.json').read_bytes())
    store=Checkpoint(cold,plan,headers,check)
    cfg=json.loads((cold/'config.json').read_bytes())['text_config'];cfg['_attn_implementation']='eager'
    config=SimpleNamespace(**cfg);official=load(ROOT/'sources',config)
    tokenizer=Tokenizer.from_file(str(cold/'tokenizer.json'))
    prompt='The capital of France is'
    ids=tokenizer.encode(prompt,add_special_tokens=False).ids
    if not 1<=len(ids)<=32 or any(not 0<=i<248320 for i in ids):raise ValueError('Frozen prompt extent')
    generation=json.loads((cold/'generation_config.json').read_bytes())
    if set(generation['eos_token_id'])!={248044,248046}:raise ValueError('Pinned generation stopping')
    cache=original_cache(official,config.layer_types);evidence=Evidence(ROOT/'evidence',check)
    position=0;step=0;generated=[];steps=[];prefill_scan_comparisons=[]

    def observe_internal(name,value):
        evidence.observe(f'step{step}-layer{evidence.context}-'+name,value)
    def attention_domains(scores,mask):
        # Only legal causal entries determine the exp domain. Diagnostics use
        # copies/casts and never feed any value back into official computation.
        values=scores.detach().float()
        valid=torch.ones_like(values,dtype=torch.bool) if mask is None else (mask==0).expand_as(values)
        if not valid.any(dim=-1).all():raise ValueError('Attention row without a valid key')
        maximum=values.masked_fill(~valid,float('-inf')).max(dim=-1,keepdim=True).values
        observe_internal('attention_valid_scores',values[valid])
        observe_internal('attention_valid_softmax_shifts',(values-maximum)[valid])
    def delta_domains(a,dt_bias,A_log,g):
        observe_internal('softplus_input',a.detach().float()+dt_bias.detach())
        observe_internal('A_log_exp_input',A_log.detach().float())
        observe_internal('recurrent_decay_exp_input',g)
    official['_observe']=observe_internal
    official['_observe_attention']=attention_domains
    official['_observe_delta_gate']=delta_domains

    def observer(name,save=False):
        def hook(module,args,output):
            value=output[0] if isinstance(output,tuple) else output
            key=f'step{step}-'+name;evidence.observe(key,value)
            if 'norm' in name and args and isinstance(args[0],torch.Tensor):
                evidence.observe(key+'-input_rms_square_mean',args[0].detach().float().pow(2).mean(-1))
            if save and isinstance(value,torch.Tensor):evidence.tensors(key,dict(output=value))
        return hook

    # Observe original function arguments/results without changing their bodies,
    # identities or numerical inputs. This records real recurrence gate domains.
    original_scan=official['torch_recurrent_gated_delta_rule']
    for function_name in ('torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule'):
        original=official[function_name]
        def wrap(*args,_original=original,_name=function_name,**kwargs):
            prefix=f'step{step}-layer{evidence.context}-'+_name
            for name in ('g','beta'):evidence.observe(prefix+'-'+name,kwargs[name])
            result=_original(*args,**kwargs)
            evidence.observe(prefix+'-output',result[0]);evidence.observe(prefix+'-state',result[1])
            if _name=='torch_chunk_gated_delta_rule':
                # A diagnostic alternative on the same actual operands, never
                # fed into the nominal model or used to narrow source bounds.
                scan_output,scan_state=original_scan(*args,**kwargs)
                output_delta=(scan_output.float()-result[0].float()).abs()
                state_delta=(scan_state-result[1]).abs()
                prefill_scan_comparisons.append(dict(step=step,layer=evidence.context,
                    BF16_output_mismatches=int((scan_output!=result[0]).sum()),
                    output_max_abs=float(output_delta.max()),state_max_abs=float(state_delta.max()),
                    scan_reference='original_torch_recurrent_gated_delta_rule_on_actual_chunk_inputs',
                    nominal_feedback_unchanged=True))
                evidence.tensors(f'step{step}-layer{evidence.context}-prefill_scan',dict(
                    query=args[0],key=args[1],value=args[2],g=kwargs['g'],beta=kwargs['beta'],
                    scan_output=scan_output,scan_state=scan_state))
            return result
        official[function_name]=wrap

    class StreamingLayer(nn.Module):
        def __init__(self,index):super().__init__();self.index=index
        def forward(self,hidden_states,**kwargs):
            check();i=self.index;evidence.context=i
            evidence.tensors(f'step{step}-layer{i}-input',dict(hidden=hidden_states))
            with torch.device('meta'):module=official['Qwen3_5DecoderLayer'](config,i)
            module=store.bind_layer(module,i)
            for name,submodule in module.named_modules():
                if name and (isinstance(submodule,nn.Linear) or 'norm' in name or name=='mlp.act_fn'):
                    submodule.register_forward_hook(observer(f'layer{i}-'+name.replace('.','_'),save=i in (0,3)))
            output=module(hidden_states,**kwargs)
            evidence.observe(f'step{step}-layer{i}-output',output)
            evidence.tensors(f'step{step}-layer{i}-output',dict(hidden=output))
            del module;gc.collect()
            if i in (19,43,63):evidence.state(step,{19:0,43:1,63:2}[i],cache,output)
            return output

    def embedding(input_ids):
        result=torch.empty((*input_ids.shape,5120),dtype=torch.bfloat16)
        for column,token in enumerate(input_ids[0].tolist()):
            result[0,column]=store.tensor('model.language_model.embed_tokens.weight',token,1)[0]
        return result

    with torch.device('meta'):norm=official['Qwen3_5RMSNorm'](5120,eps=config.rms_norm_eps)
    norm.weight=nn.Parameter(store.tensor('model.language_model.norm.weight'),requires_grad=False)
    model=SimpleNamespace(config=config,embed_tokens=embedding,layers=[StreamingLayer(i) for i in range(64)],
        rotary_emb=official['Qwen3_5TextRotaryEmbedding'](config),norm=norm)
    started=time.monotonic()
    with torch.inference_mode():
        for step in range(4):
            check();input_ids=torch.tensor([ids],dtype=torch.long);length=len(ids)
            # No padding: explicit eager causal mask, passed through the original
            # text forward's documented per-layer-type mask mapping entry point.
            q=torch.arange(position,position+length)[:,None];k=torch.arange(position+length)[None,:]
            mask=torch.zeros((length,position+length),dtype=torch.bfloat16)
            mask.masked_fill_(k>q,torch.finfo(torch.bfloat16).min)
            masks={'full_attention':mask[None,None,:,:],'linear_attention':None}
            before=cache.get_seq_length()
            if before!=position:raise ValueError('Official cache position mismatch')
            output=official['text_model_forward'](model,input_ids=input_ids,past_key_values=cache,
                attention_mask=masks,use_cache=True)
            hidden=output.last_hidden_state[:,-1:,:]
            evidence.tensors(f'step{step}-final_norm',dict(hidden=hidden))
            # This is nn.Linear.forward's unchanged operation, using the complete
            # original vocabulary matrix at once within the finite RAM budget.
            head=store.tensor('lm_head.weight');logits=torch.nn.functional.linear(hidden,head)
            evidence.observe(f'step{step}-logits',logits);evidence.tensors(f'step{step}-logits',dict(logits=logits))
            token=int(torch.argmax(logits[0,-1]).item());generated.append(token)
            ranked=torch.topk(logits[0,-1].float(),2)
            position+=length
            if cache.get_seq_length()!=position:raise ValueError('Cache valid length did not advance')
            steps.append(dict(step=step,input_ids=ids,position_after=position,next_token=token,
                recurrent_path='official_chunk_prefill' if step==0 else 'official_incremental_decode',
                input_positions=list(range(position-length,position)),valid_context=position,
                top_two_ids=ranked.indices.tolist(),top_two_raw_logits=ranked.values.tolist(),
                raw_logit_top_gap=float(ranked.values[0]-ranked.values[1]),
                maximum_ties=int((logits[0,-1]==logits[0,-1,token]).sum().item()),
                state_dtypes=dict(conv='bfloat16',recurrent='float32',KV='bfloat16')))
            del head,logits,output;gc.collect();store.check_unchanged()
            if token in generation['eos_token_id']:break
            if step<3:
                cache,restore_receipt=restore(official,config.layer_types,evidence,step,cache,position,check)
                steps[-1]['restore_before_next_step']=restore_receipt
            ids=[token]
    if store.read_names!=set(store.tensors):raise ValueError('Not all851 original tensor names used')
    result=dict(status='nominal_reference_complete',hardware_acceptance=False,
        prompt=prompt,prompt_format='raw_text_no_chat_template',add_special_tokens=False,
        generated_ids=generated,generated_text=tokenizer.decode(generated,skip_special_tokens=False),
        generation_override=dict(do_sample=False,max_new_tokens=4,eos_token_id=generation['eos_token_id']),
        steps=steps,source_extraction=official['extraction'],text_tensors_bound=851,
        prefill_scan_comparisons=prefill_scan_comparisons,
        files=evidence.files,ranges=evidence.ranges,seconds=time.monotonic()-started,
        note='Nominal official arithmetic observations. Device numerical contracts and complete-model acceptance remain separate.')
    with (ROOT/'reference.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')


if __name__=='__main__':
    raise SystemExit('Run only through the exact bounded reference entry; this draft has not been executed.')
