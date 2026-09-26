"""Capture original-model router/SwiGLU/join operands without changing arithmetic."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'reference'),str(ROOT/'core')]
from run_reference import Checkpoint,ModelConfig,mlp,from_bits
import gpt_oss.torch.model as upstream


def bits(t):
    return t.contiguous().view(torch.int16).numpy().astype(np.uint16,copy=True)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    module=mlp(Checkpoint(args.model),ModelConfig(num_hidden_layers=24,num_experts=32),0)
    captured={}
    def gate_hook(module,args,out):captured['logits']=out.clone()
    handle=module.gate.register_forward_hook(gate_hook)
    original_swiglu=upstream.swiglu
    original_einsum=torch.einsum
    def swiglu_hook(x,*a,**kw):
        captured['swiglu_input']=x.clone()
        out=original_swiglu(x,*a,**kw)
        captured['swiglu_output']=out.clone()
        return out
    def einsum_hook(equation,*operands):
        out=original_einsum(equation,*operands)
        if equation=='bec,be->bc':
            captured['expert_outputs']=operands[0].clone()
            captured['probabilities']=operands[1].clone()
            captured['joined']=out.clone()
        return out
    try:
        upstream.swiglu=swiglu_hook;torch.einsum=einsum_hook
        x=from_bits(np.load(args.reference/'step00-layer00-attention.npy',allow_pickle=False))
        result=module(x)
    finally:
        upstream.swiglu=original_swiglu;torch.einsum=original_einsum;handle.remove()
    expected=np.load(args.reference/'step00-layer00-output.npy',allow_pickle=False)
    if not np.array_equal(bits(result),expected):raise ValueError('Instrumented original forward changed')
    real_logits=captured['logits'][0]
    logit_cases=torch.stack([real_logits,torch.arange(32,dtype=torch.float32).to(torch.bfloat16),
                             torch.linspace(-20,20,32).to(torch.bfloat16)])
    chosen=torch.topk(logit_cases,4,dim=-1,sorted=True)
    if torch.any(torch.sort(real_logits,descending=True).values[3:4]==torch.sort(real_logits,descending=True).values[4:5]):
        raise ValueError('Real fixture has an ambiguous top-four boundary')
    probability_cases=torch.softmax(chosen.values,dim=-1)
    interleaved=captured['swiglu_input'][0,0,:512].clone()
    # Exercise clamps, negatives, large magnitudes, and the +1 branch explicitly.
    edge=torch.tensor([[-30,20],[-7,-20],[-1,-1],[0,0],[1,1],[7,7],[20,20],[.5,-.5]],dtype=torch.bfloat16).reshape(-1)
    interleaved[-len(edge):]=edge
    activation=original_swiglu(interleaved)
    expert_outputs=captured['expert_outputs'][0,:,:160].clone()
    probabilities=captured['probabilities'][0].clone()
    joined=original_einsum('ec,e->c',expert_outputs,probabilities)
    path=args.output/'math-fixture.npz'
    with path.open('xb') as out:
        np.savez(out,logits=bits(logit_cases),ids=chosen.indices.numpy().astype(np.uint16),
                 probabilities=bits(probability_cases),interleaved=bits(interleaved),activation=bits(activation),
                 experts=bits(expert_outputs),join_probabilities=bits(probabilities),joined=bits(joined))
    report=dict(model='openai/gpt-oss-20b',layer=0,real_forward_reproduced_bitwise=True,
                source_reference=str(args.reference),fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                real_router_top4=chosen.indices[0].tolist(),top4_tie_policy='ascending expert ID for exact ties in CSL',
                nonlinear_absolute_exp_tail_bound=1.81e-35,full_model=False)
    (args.output/'math-fixture.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
