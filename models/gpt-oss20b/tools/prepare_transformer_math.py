"""Capture original norm/RoPE/attention operands for Hello followed by comma."""
import argparse,hashlib,json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'reference'),str(ROOT/'core')]
from run_reference import Checkpoint,ModelConfig,attention,from_bits
import gpt_oss.torch.model as upstream

def bits(t):return t.contiguous().view(torch.int16).numpy().astype(np.uint16,copy=True)

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    c=Checkpoint(args.model);tokens=[13225,11]
    module=attention(c,ModelConfig(num_hidden_layers=24,num_experts=32),0)
    hidden=from_bits(c.tensor('model.embed_tokens.weight')[tokens])
    captured={}
    h1=module.norm.register_forward_hook(lambda m,a,o:captured.update(normalized=o.clone()))
    h2=module.qkv.register_forward_hook(lambda m,a,o:captured.update(qkv=o.clone()))
    original=upstream.sdpa
    def wrapper(Q,K,V,S,scale,sliding_window=0):
        out=original(Q,K,V,S,scale,sliding_window)
        captured.update(Q=Q.clone(),K=K.clone(),V=V.clone(),sinks=S.clone(),attention=out.clone())
        alternative=original(Q,K,V,torch.full_like(S,-float('inf')),scale,sliding_window)
        captured['sink_effect']=float((out.float()-alternative.float()).abs().max())
        return out
    try:
        upstream.sdpa=wrapper
        result=module(hidden)
    finally:
        upstream.sdpa=original;h1.remove();h2.remove()
    expected=np.load(args.reference/'step01-layer00-attention.npy',allow_pickle=False)
    if not np.array_equal(bits(result),expected):raise ValueError('Instrumented attention changed original forward')
    concentration,frequencies=module.rope._compute_concentration_and_inv_freq()
    args.output.mkdir(parents=True,exist_ok=False)
    path=args.output/'transformer-fixture.npz'
    with path.open('xb') as out:
        np.savez(out,hidden=bits(hidden),gain=np.array(c.tensor('model.layers.0.input_layernorm.weight')),
            normalized=bits(captured['normalized']),frequencies=frequencies.numpy(),
            raw_q=bits(captured['qkv'][:,:64]),raw_k=bits(captured['qkv'][:,4096:4160]),
            rotary_q=bits(captured['Q'][:,0,0,:]),rotary_k=bits(captured['K'][:,0,:]),
            queries=bits(captured['Q'][:,0,:,:].reshape(2,512)),keys=bits(captured['K'][:,0,:]),
            values=bits(captured['V'][:,0,:]),sinks=bits(captured['sinks'][:8]),
            attention=bits(captured['attention'][:,:512]))
    report=dict(model='openai/gpt-oss-20b',layer=0,kv_head=0,query_heads=list(range(8)),tokens=tokens,
                original_attention_reproduced_bitwise=True,learned_sink_max_effect=captured['sink_effect'],
                yarn_concentration=concentration,fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),full_model=False)
    assert report['learned_sink_max_effect']>0
    (args.output/'transformer-fixture.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
