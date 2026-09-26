"""Real full-width 160x2880 expert projection, distributed across ten weight PEs."""
import argparse,hashlib,json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'reference'),str(ROOT/'core')]
from run_reference import Checkpoint,ModelConfig,mlp,from_bits
from checkpoint import decode_mxfp4

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert json.loads((args.model/'scale-audit.json').read_text())['passed']
    c=Checkpoint(args.model)
    record=json.loads((args.reference/'layers.jsonl').open().readline())
    expert=record['routing'][0]['experts'][0][0]
    prefix='model.layers.0.mlp.experts.gate_up_proj'
    blocks=np.array(c.tensor(prefix+'_blocks')[expert,:160],copy=True)
    scales=np.array(c.tensor(prefix+'_scales')[expert,:160],copy=True)
    matrix=decode_mxfp4(blocks,scales).reshape(160,2880)
    module=mlp(c,ModelConfig(num_hidden_layers=24,num_experts=32),0)
    vectors=[]
    for step in (0,1):
        hidden=from_bits(np.load(args.reference/f'step{step:02}-layer00-attention.npy',allow_pickle=False))[-1:]
        vectors.append(module.norm(hidden)[0].float().numpy())
    vectors=np.stack([*vectors,np.zeros(2880,np.float32)])
    exact=vectors.astype(np.float64)@matrix.astype(np.float64).T
    sums=np.abs(vectors.astype(np.float64))@np.abs(matrix.astype(np.float64)).T
    operations=2*2880+20
    gamma=(operations*2**-24)/(1-operations*2**-24)
    bounds=gamma*sums+np.finfo(np.float32).tiny
    tiles=np.ascontiguousarray(blocks.reshape(160,10,9,16).transpose(1,0,2,3))
    scale_tiles=np.ascontiguousarray(scales.reshape(160,10,9).transpose(1,0,2))
    args.output.mkdir(parents=True,exist_ok=False)
    path=args.output/'fixture.npz'
    with path.open('xb') as out:np.savez(out,blocks=tiles,scales=scale_tiles,vectors=vectors,exact=exact,bounds=bounds)
    meta=dict(model='openai/gpt-oss-20b',revision='6cee5e81ee83917806bbde320786a8fb61efebee',layer=0,expert=expert,
              rows=[0,160],columns=[0,2880],weight_pes=10,packed_weight_bytes=int(tiles.nbytes+scale_tiles.nbytes),
              fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              scope='full hidden-dimension gate/up projection before bias; two real normalized hidden states plus zero',
              numerical_policy='FP64 dot oracle; gamma(5780) absolute summation bound',full_model=False)
    (args.output/'fixture.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta))

if __name__=='__main__':main()
