"""All original layer-zero router weights, norm gain and two real hidden states."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'reference'),str(ROOT/'core')]
from run_reference import Checkpoint,ModelConfig,mlp,from_bits
def bits(t):return t.detach().contiguous().view(torch.int16).numpy().view(np.uint16).copy()
@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    c=Checkpoint(a.model);module=mlp(c,ModelConfig(num_hidden_layers=24,num_experts=32),0)
    fields={k:[] for k in ('hidden','normalized','logits','ids','probabilities')}
    for step in (0,1):
        hidden=from_bits(np.load(a.reference/f'step{step:02}-layer00-attention.npy',allow_pickle=False))[-1:]
        norm=module.norm(hidden);logits=module.gate(norm);chosen=torch.topk(logits,4,dim=-1,sorted=True)
        for key,value in [('hidden',hidden[0]),('normalized',norm[0]),('logits',logits[0]),('probabilities',torch.softmax(chosen.values,dim=1)[0])]:
            fields[key].append(bits(value))
        fields['ids'].append(chosen.indices[0].numpy().astype(np.uint16))
    arrays={k:np.stack(v) for k,v in fields.items()}
    arrays['weights']=np.ascontiguousarray(c.tensor('model.layers.0.mlp.router.weight').reshape(32,30,96).transpose(1,2,0))
    arrays['bias']=np.array(c.tensor('model.layers.0.mlp.router.bias'),copy=True)
    arrays['gain']=np.array(c.tensor('model.layers.0.post_attention_layernorm.weight'),copy=True)
    a.output.mkdir(parents=True,exist_ok=False);path=a.output/'router-fixture.npz'
    with path.open('xb') as out:np.savez(out,**arrays)
    meta=dict(model='openai/gpt-oss-20b',layer=0,fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              full_model=False,scope='complete original 2880 RMSNorm, 32x2880 router projection and top4 probabilities',
              numerical_policy='exact original BF16 normalized inputs, logits, top4 IDs and probabilities')
    (a.output/'router-fixture.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))
if __name__=='__main__':main()
