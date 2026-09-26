"""Localize position-zero differences using actual physical layer inputs.

The original operator replay and a naive-FP32-RMSNorm diagnostic are kept
separate. Neither replaces the unchanged full-prefix acceptance criteria.
"""
import argparse,ast,ctypes,gc,hashlib,json,sys,time,types
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn

p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);a=p.parse_args();base=a.reference
sys.path.insert(0,str(base))
from pinned import load,original_cache
from checkpoint import Checkpoint
torch.set_num_threads(2);torch.set_num_interop_threads(1)
tree=ast.parse((base/'run.py').read_text());ns=dict(np=np)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['expand','metrics']],type_ignores=[]),'frozen_metrics','exec'),ns)
criteria=json.loads((base/'full-acceptance-v1.json').read_text())
model=Path('/srv/model-storage/qwen38-singlewse/model')
cfg=json.loads((model/'config.json').read_text())['text_config'];cfg['_attn_implementation']='eager';config=SimpleNamespace(**cfg)
official=load(base/'sources',config,observe=False);store=Checkpoint(model,base/'tensors.json')
generation=json.loads((base/'candidate/generation.json').read_text());positions=generation['processed_positions']
candidate=np.memmap(base/'candidate/layers.bf16',dtype='<u2',mode='r',shape=(positions,64,5120))
rotary=official['Qwen3_5TextRotaryEmbedding'](config);pos=torch.tensor([[0]])
records=[];started=time.monotonic()

def naive_norm(module,x):
    values=x.detach().float().numpy();total=np.zeros(values.shape[:-1],np.float32)
    for i in range(values.shape[-1]):total=np.float32(total+np.float32(values[...,i]*values[...,i]))
    inverse=np.float32(1)/np.sqrt(np.float32(total/np.float32(values.shape[-1])+np.float32(module.eps)))
    gain=np.float32(1)+module.weight.detach().float().numpy()
    out=np.float32(np.float32(values*inverse[...,None])*gain)
    return torch.from_numpy(out.copy()).to(x.dtype)

with torch.inference_mode():
 for layer in range(64):
    tick=time.monotonic()
    if layer==0:hidden=store.tensor('model.language_model.embed_tokens.weight',generation['prompt_ids'][0],1)[None]
    else:hidden=torch.from_numpy(np.array(candidate[0,layer-1],copy=True)).view(torch.bfloat16).reshape(1,1,5120)
    with torch.device('meta'):module=official['Qwen3_5DecoderLayer'](config,layer)
    module=store.bind_layer(module,layer);embeddings=rotary(hidden,pos[None].expand(3,-1,-1))
    variants={}
    for variant in ['original','naive_rmsnorm_diagnostic']:
      if variant!='original':
        for norm in [module.input_layernorm,module.post_attention_layernorm]:norm.forward=types.MethodType(naive_norm,norm)
      cache=original_cache(official,config.layer_types)
      mask=torch.zeros((1,1,1,1),dtype=torch.bfloat16) if config.layer_types[layer]=='full_attention' else None
      out=module(hidden,position_embeddings=embeddings,attention_mask=mask,position_ids=pos,past_key_values=cache)
      bits=out.view(torch.uint16).numpy().reshape(-1)
      np.save(f'layer{layer}-{variant}.npy',bits)
      metrics=ns['metrics'](candidate[0,layer],bits,criteria['per_position_all64_layers_and_final_norm'])
      metrics['bit_exact_values']=int(np.count_nonzero(candidate[0,layer]==bits));variants[variant]=metrics
      del out,cache
    record=dict(layer=layer,position=0,actual_physical_layer_input=True,variants=variants,seconds=time.monotonic()-tick)
    records.append(record);Path('progress.json').write_text(json.dumps(dict(layers=len(records),last=record,seconds=time.monotonic()-started))+'\n')
    print(json.dumps(record),flush=True);del module,hidden;gc.collect();ctypes.CDLL(None).malloc_trim(0)
store.check_unchanged()
Path('COMPLETE.json').write_text(json.dumps(dict(diagnostic=True,full_model_accepted=False,criteria_changed=False,positions=[0],records=records,seconds=time.monotonic()-started),indent=2)+'\n')
