"""Original full layer-0 attention weights and two causal upstream input/output pairs."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'reference'),str(ROOT/'core')]
from run_reference import Checkpoint,ModelConfig,attention,from_bits
import gpt_oss.torch.model as upstream

def bits(t):return t.detach().contiguous().view(torch.int16).numpy().view(np.uint16).copy()
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    c=Checkpoint(a.model);module=attention(c,ModelConfig(num_hidden_layers=24,num_experts=32),0)
    hidden=from_bits(c.tensor('model.embed_tokens.weight')[[13225,11]])
    fields={};h1=module.norm.register_forward_hook(lambda m,x,y:fields.update(normalized=bits(y)))
    h2=module.qkv.register_forward_hook(lambda m,x,y:fields.update(qkv=bits(y)))
    original=upstream.sdpa
    def capture(Q,K,V,S,scale,sliding_window=0):
        result=original(Q,K,V,S,scale,sliding_window)
        fields.update(rotary_q=bits(Q.reshape(2,4096)),keys=bits(K),values=bits(V),attention=bits(result))
        return result
    try:upstream.sdpa=capture;expected=bits(module(hidden))
    finally:upstream.sdpa=original;h1.remove();h2.remove()
    saved=np.load(a.reference/'step01-layer00-attention.npy',allow_pickle=False)
    if not np.array_equal(saved,expected):raise ValueError('Original complete-prefix oracle mismatch')
    if not np.array_equal(expected[:1],np.load(a.reference/'step00-layer00-attention.npy',allow_pickle=False)):
        raise ValueError('First causal output differs between complete-prefix reference runs')
    _,freq=module.rope._compute_concentration_and_inv_freq()
    qkv=np.concatenate([c.tensor('model.layers.0.self_attn.'+k+'_proj.weight') for k in ('q','k','v')],axis=0)
    qb=np.concatenate([c.tensor('model.layers.0.self_attn.'+k+'_proj.bias') for k in ('q','k','v')])
    padded=np.zeros((2944,4128),np.uint16);padded[:2880,:4096]=c.tensor('model.layers.0.self_attn.o_proj.weight')
    ob=np.zeros(2944,np.uint16);ob[:2880]=c.tensor('model.layers.0.self_attn.o_proj.bias')
    arrays=dict(hidden=bits(hidden),expected_output=expected,frequencies=freq.numpy(),
      gain=np.array(c.tensor('model.layers.0.input_layernorm.weight'),copy=True),
      sinks=np.array(c.tensor('model.layers.0.self_attn.sinks'),copy=True).reshape(8,8),
      qkv_weights=np.ascontiguousarray(qkv.reshape(40,128,30,96).transpose(0,2,3,1)),qkv_bias=qb.reshape(40,128),
      o_weights=np.ascontiguousarray(padded.reshape(23,128,43,96).transpose(0,2,3,1)),o_bias=ob.reshape(23,128),**fields)
    a.output.mkdir(parents=True,exist_ok=False);path=a.output/'attention-fixture.npz'
    with path.open('xb') as f:np.savez(f,**arrays)
    meta=dict(model='openai/gpt-oss-20b',revision='6cee5e81ee83917806bbde320786a8fb61efebee',layer=0,
      tokens=[13225,11],full_attention=True,full_model=False,application_pes=4640,
      fixture_sha256=digest(path),bytes=path.stat().st_size,
      scope='complete original layer-0 RMSNorm, QKV, YaRN, 64-query/8-KV-head attention with learned sinks, O projection and residual; persistent 96-token cache',
      numerical_policy='at most one BF16 ULP versus complete original attention; no host neural intermediates')
    (a.output/'attention-fixture.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)

if __name__=='__main__':main()
