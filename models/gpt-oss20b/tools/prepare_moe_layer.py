"""All 32 original experts, original router, and two saved whole-layer oracles."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'))
from checkpoint import Checkpoint

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''): h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--router',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    checkpoint=Checkpoint(a.model)
    receipt=json.loads((a.model/'COMPLETE.json').read_text())
    assert receipt['revision']=='6cee5e81ee83917806bbde320786a8fb61efebee' and receipt['tensors']==459
    rm=json.loads((a.router/'router-fixture.json').read_text())
    assert digest(a.router/'router-fixture.npz')==rm['fixture_sha256']
    with np.load(a.router/'router-fixture.npz',allow_pickle=False) as f:
        arrays={('router_'+k if k in ('weights','bias') else k):f[k] for k in f.files}
    arrays['expected_output']=np.stack([np.load(a.reference/f'step{s:02}-layer00-output.npy',allow_pickle=False)[-1] for s in (0,1)])
    for kind,projection,rows in [('gate','gate_up_proj',36),('down','down_proj',18)]:
        prefix='model.layers.0.mlp.experts.'+projection
        blocks=checkpoint.tensor(prefix+'_blocks');scales=checkpoint.tensor(prefix+'_scales')
        arrays[kind+'_blocks']=np.ascontiguousarray(blocks.reshape(32,rows,160,10,9,16).transpose(0,1,3,2,4,5))
        arrays[kind+'_scales']=np.ascontiguousarray(scales.reshape(32,rows,160,10,9).transpose(0,1,3,2,4))
        arrays[kind+'_bias']=np.array(checkpoint.tensor(prefix+'_bias'),copy=True).reshape(32,rows,160)
    assert arrays['expected_output'].shape==arrays['hidden'].shape==(2,2880)
    a.output.mkdir(parents=True,exist_ok=False);path=a.output/'moe-fixture.npz'
    with path.open('xb') as f: np.savez(f,**arrays)
    meta=dict(model=receipt['model'],revision=receipt['revision'],layer=0,experts=32,top_k=4,
      application_pes=31320,full_moe_layer=True,full_model=False,fixture_sha256=digest(path),
      bytes=path.stat().st_size,scope='complete original layer-0 MoE: RMSNorm, router, all 32 resident experts, top4 weighted join and residual',
      expected_ids=arrays['ids'].tolist(),numerical_policy='exact routing and normalization; at most one BF16 ULP in final layer output')
    (a.output/'moe-fixture.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)

if __name__=='__main__':main()
