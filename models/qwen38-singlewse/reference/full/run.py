"""Complete independent text reference; never physical inference evidence."""
import argparse,ctypes,gc,hashlib,json,platform,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from tokenizers import Tokenizer
from jinja2.sandbox import ImmutableSandboxedEnvironment
from pinned import load,original_cache
from checkpoint import Checkpoint

ROOT=Path.cwd();MODEL=Path('/srv/model-storage/qwen38-singlewse/model')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--qualify',action='store_true');a=parser.parse_args()
    assert torch.__version__=='2.6.0+cpu' and np.__version__=='2.2.6'
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    pins=json.loads((MODEL/'COMPLETE.json').read_text())
    for name in ['config.json','tokenizer.json','chat_template.jinja','generation_config.json']:
        pin=next(x for x in pins['files'] if x['file']==name);raw=(MODEL/name).read_bytes()
        h=hashlib.sha256(raw).hexdigest() if 'sha256' in pin else hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
        assert h==pin.get('sha256',pin.get('git_blob_sha1'))
    cfg=json.loads((MODEL/'config.json').read_text())['text_config'];cfg['_attn_implementation']='eager'
    config=SimpleNamespace(**cfg);official=load(ROOT/'sources',config,observe=False)
    workload=json.loads((ROOT/'workload.json').read_text())
    env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True,extensions=['jinja2.ext.loopcontrols'])
    def raise_exception(message):raise ValueError(message)
    env.globals['raise_exception']=raise_exception
    env.filters['tojson']=lambda x,**kw:json.dumps(x,ensure_ascii=False,**kw)
    prompt=env.from_string((MODEL/'chat_template.jinja').read_text()).render(**workload,tools=None)
    tokenizer=Tokenizer.from_file(str(MODEL/'tokenizer.json'))
    ids=tokenizer.encode(prompt,add_special_tokens=False).ids
    assert 0<len(ids)<=workload['max_context']-workload['max_new_tokens']
    source_record=dict(extraction=official['extraction'],prompt=prompt,input_ids=ids,workload=workload,
        torch=torch.__version__,numpy=np.__version__,python=platform.python_version())
    with (ROOT/'qualification.json').open('x') as f:json.dump(source_record,f,indent=2);f.write('\n')
    with torch.device('meta'):
        for index in [0,3]:
            module=official['Qwen3_5DecoderLayer'](config,index)
            expected={n.removeprefix(f'model.language_model.layers.{index}.') for n in json.loads((ROOT/'tensors.json').read_text())['tensors']
                if n.startswith(f'model.language_model.layers.{index}.') and not n.endswith('_scale_inv')}
            assert set(dict(module.named_parameters()))==expected
    cache=original_cache(official,config.layer_types)
    rotary=official['Qwen3_5TextRotaryEmbedding'](config)
    assert cache.get_seq_length()==0 and rotary.inv_freq.numel()==32
    if a.qualify:
        print(json.dumps(dict(qualified=True,prompt_tokens=len(ids),max_new_tokens=workload['max_new_tokens'],full_reference_executed=False)),flush=True);return
    store=Checkpoint(MODEL,ROOT/'tensors.json');evidence=ROOT/'evidence';evidence.mkdir()
    step=0
    class StreamingLayer(nn.Module):
        def __init__(self,index):super().__init__();self.index=index
        def forward(self,hidden_states,**kwargs):
            begin=time.monotonic()
            with torch.device('meta'):module=official['Qwen3_5DecoderLayer'](config,self.index)
            module=store.bind_layer(module,self.index)
            if self.index in [0,3]:
                np.save(evidence/f'step{step}-layer{self.index}-input.npy',hidden_states.view(torch.uint16).cpu().numpy())
            output=module(hidden_states,**kwargs)
            if not torch.isfinite(output).all():raise ValueError('Nonfinite reference hidden state')
            np.save(evidence/f'step{step}-layer{self.index}-output.npy',output.view(torch.uint16).cpu().numpy())
            del module;gc.collect()
            ctypes.CDLL(None).malloc_trim(0)
            print(json.dumps(dict(step=step,layer=self.index,seconds=time.monotonic()-begin,max_abs=float(output.float().abs().max()))),flush=True)
            return output
    def embedding(tokens):
        return torch.stack([store.tensor('model.language_model.embed_tokens.weight',t,1)[0] for t in tokens[0].tolist()])[None]
    with torch.device('meta'):norm=official['Qwen3_5RMSNorm'](5120,eps=config.rms_norm_eps)
    norm.weight=nn.Parameter(store.tensor('model.language_model.norm.weight'),requires_grad=False)
    model=SimpleNamespace(config=config,embed_tokens=embedding,layers=[StreamingLayer(i) for i in range(64)],rotary_emb=rotary,norm=norm)
    eos=json.loads((MODEL/'generation_config.json').read_text())['eos_token_id'];generated=[];steps=[];position=0;started=time.monotonic()
    with torch.inference_mode():
        for step in range(workload['max_new_tokens']):
            begin=time.monotonic();length=len(ids)
            assert position+length<=workload['max_context'] and cache.get_seq_length()==position
            mask=torch.zeros((length,position+length),dtype=torch.bfloat16)
            mask.masked_fill_(torch.arange(position+length)[None,:]>torch.arange(position,position+length)[:,None],torch.finfo(torch.bfloat16).min)
            output=official['text_model_forward'](model,input_ids=torch.tensor([ids]),past_key_values=cache,
                attention_mask={'full_attention':mask[None,None],'linear_attention':None},use_cache=True)
            hidden=output.last_hidden_state[:,-1:,:]
            # Full original BF16 head, bounded row tiles, every vocabulary entry.
            logits=torch.empty(248320,dtype=torch.bfloat16)
            for start in range(0,248320,1024):
                count=min(1024,248320-start)
                head=store.tensor('lm_head.weight',start,count)
                logits[start:start+count]=torch.nn.functional.linear(hidden,head).reshape(-1)
            assert torch.isfinite(logits).all()
            np.save(evidence/f'step{step}-logits.npy',logits.view(torch.uint16).numpy())
            token=int(torch.argmax(logits));generated.append(token);position+=length
            assert cache.get_seq_length()==position
            top=torch.topk(logits.float(),2)
            record=dict(step=step,input_ids=ids,next_token=token,position=position,top_ids=top.indices.tolist(),top_logits=top.values.tolist(),seconds=time.monotonic()-begin)
            steps.append(record);print(json.dumps(dict(record=record,text=tokenizer.decode(generated,skip_special_tokens=False))),flush=True)
            store.check_unchanged()
            (ROOT/'progress.json').write_text(json.dumps(dict(steps=steps,generated_ids=generated,text=tokenizer.decode(generated,skip_special_tokens=False),physical=False),indent=2)+'\n')
            if token in eos:break
            ids=[token]
        assert store.read_names==set(store.tensors)
    result=dict(complete=True,physical=False,full_model_layers=64,text_tensors_read=len(store.read_names),generated_ids=generated,
        generated_text=tokenizer.decode(generated,skip_special_tokens=False),eos_reached=generated[-1] in eos,steps=steps,
        seconds=time.monotonic()-started,scope='Independent explicit CPU FP8 numerical reference; not CS-3 latency or physical acceptance')
    (ROOT/'COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
