"""Build a bounded all-layer RMSNorm accuracy fixture on remote storage."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from checkpoint import Checkpoint,bf16_float,float_bf16

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    c=Checkpoint(a.model);rows=[];gains=[];labels=[]
    with np.load(a.capture,allow_pickle=False) as f:actual={k:f[k] for k in ('attention','output')}
    for source in ('original-0','original-1','device-0'):
        step=int(source[-1]);device=source.startswith('device')
        for layer in range(24):
            for kind in ('input','post_attention'):
                if kind=='input' and layer==0:raw=np.array(c.row_slice('model.embed_tokens.weight',[13225,11][step])[0])
                else:
                    which='output' if kind=='input' else 'attention';index=layer-1 if kind=='input' else layer
                    raw=actual[which][index] if device else np.load(a.reference/f'step{step:02}-layer{index:02}-{which}.npy',allow_pickle=False)[-1]
                rows.append(raw);gains.append(np.array(c.tensor(f'model.layers.{layer}.{kind}_layernorm.weight')))
                labels.append(dict(source=source,layer=layer,kind=kind))
    hidden=np.stack(rows);gain=np.stack(gains);x=bf16_float(hidden);g=bf16_float(gain)
    d=x.astype(np.float64);gold=d/np.sqrt(np.mean(d*d,axis=1,keepdims=True)+1e-5)*g.astype(np.float64)
    # Quantize FP64 on the exact BF16 lattice before converting to FP32, avoiding
    # double rounding at a BF16 midpoint. numpy.rint uses ties-to-even.
    quantum=np.ldexp(np.ones_like(gold),np.maximum(np.frexp(np.abs(gold))[1]-8,-133))
    rounded=np.copysign(np.rint(np.abs(gold)/quantum)*quantum,gold)
    fp64=float_bf16(rounded.astype(np.float32))
    t=torch.from_numpy(x);sc=torch.from_numpy(g)
    expected=(t*torch.rsqrt(torch.mean(t*t,dim=1,keepdim=True)+1e-5)*sc).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    a.output.mkdir(parents=True,exist_ok=False);path=a.output/'norm-fixture.npz'
    np.savez(path,hidden=hidden,gain=gain,fp64=fp64,torch=expected)
    meta=dict(cases=len(hidden),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),labels=labels,
      scope='original all-layer BF16 activations and gains; FP64 RMSNorm and original FP32 Torch references')
    (a.output/'norm-fixture.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(dict(cases=len(hidden),sha256=meta['sha256'])))

if __name__=='__main__':main()
