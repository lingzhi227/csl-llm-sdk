"""Original expert weights and two independent upstream BF16 expert oracles."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'reference'),str(ROOT/'core')]
from run_reference import Checkpoint,ModelConfig,mlp,from_bits
from checkpoint import decode_mxfp4
from gpt_oss.torch.model import swiglu

def bits(x):return x.detach().contiguous().view(torch.int16).numpy().view(np.uint16).copy()

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--full',action='store_true')
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert json.loads((a.model/'scale-audit.json').read_text())['passed']
    c=Checkpoint(a.model);expert=5;layer=0
    gr,dr=(36,18) if a.full else (2,1)
    arrays={};matrices={};biases={}
    for kind,projection,rows in [('gate','gate_up_proj',gr),('down','down_proj',dr)]:
        prefix=f'model.layers.{layer}.mlp.experts.{projection}'
        b=np.array(c.tensor(prefix+'_blocks')[expert,:rows*160],copy=True)
        s=np.array(c.tensor(prefix+'_scales')[expert,:rows*160],copy=True)
        bias=np.array(c.tensor(prefix+'_bias')[expert,:rows*160],copy=True)
        arrays[kind+'_blocks']=np.ascontiguousarray(b.reshape(rows,160,10,9,16).transpose(0,2,1,3,4))
        arrays[kind+'_scales']=np.ascontiguousarray(s.reshape(rows,160,10,9).transpose(0,2,1,3))
        arrays[kind+'_bias']=bias.reshape(rows,160)
        matrices[kind]=torch.from_numpy(decode_mxfp4(b,s).reshape(rows*160,2880)).to(torch.bfloat16)
        biases[kind]=from_bits(bias)
    module=mlp(c,ModelConfig(num_hidden_layers=24,num_experts=32),layer)
    vectors=[];gates=[];acts=[];outputs=[]
    for step in (0,1):
        hidden=from_bits(np.load(a.reference/f'step{step:02}-layer00-attention.npy',allow_pickle=False))[-1:]
        vector=module.norm(hidden)
        gate=torch.einsum('ck,bk->bc',matrices['gate'],vector)+biases['gate']
        activation=torch.zeros((1,2880),dtype=torch.bfloat16)
        activation[:,:gr*80]=swiglu(gate)
        out=torch.einsum('ck,bk->bc',matrices['down'],activation)+biases['down']
        vectors.append(vector[0].float().numpy());gates.append(bits(gate[0]))
        acts.append(bits(activation[0]));outputs.append(bits(out[0]))
    arrays.update(vectors=np.stack(vectors),expected_gate=np.stack(gates),expected_activation=np.stack(acts),expected_output=np.stack(outputs))
    a.output.mkdir(parents=True,exist_ok=False);path=a.output/'expert-fixture.npz'
    with path.open('xb') as f:np.savez(f,**arrays)
    meta=dict(model='openai/gpt-oss-20b',revision='6cee5e81ee83917806bbde320786a8fb61efebee',layer=layer,expert=expert,
              gate_rows=gr,down_rows=dr,full_expert=a.full,full_model=False,application_pes=23*gr,
              fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              scope='complete original expert' if a.full else 'protocol fragment: 320 gate/up rows, 160 activated values zero-padded to 2880, 160 down rows',
              numerical_policy='at most one BF16 ULP against original PyTorch expert operations; finite values required')
    (a.output/'expert-fixture.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))

if __name__=='__main__':main()
