"""Reuse frozen P6 PyTorch groups and bind a pinned original Qwen weight tile."""
import hashlib,json
from pathlib import Path
import numpy as np
from weights import OriginalWeights
qmeta=json.loads(Path('quant-fixture.json').read_text());assert hashlib.sha256(Path('quant-fixture.npz').read_bytes()).hexdigest()==qmeta['fixture_sha256']
with np.load('quant-fixture.npz',allow_pickle=False) as f:arrays={n:f[n] for n in ['groups','quant','scales']}
model=Path('/srv/model-storage/qwen38-singlewse/model')
if not model.exists():model=Path('/srv/qwen38-singlewse-hardware/model')
meta=json.loads(Path('tensors.json').read_text());assert meta['revision']=='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a'
pin={'layers-0.safetensors':'07f700e293baeaf3cd4240c3df1a948c4403f16961ea7979e86c8d6a9f8fd466'}
reader=OriginalWeights(model,meta['tensors'],pin);name='model.language_model.layers.0.mlp.gate_proj.weight'
codes=reader.rows(name,0,2)[:,:128].copy();assert np.all((codes&127)!=127)
raw_scale=reader.rows(name+'_scale_inv',0,1)[0,0];weight_scale=(np.array([raw_scale],np.uint32)<<16).view(np.float32)
weights=np.ascontiguousarray(codes.T).view('<u2').reshape(-1)
def decode(v):
 c=v.astype(np.int32);e=(c&127)>>3;m=c&7
 out=np.where(e==0,m*2.0**-9,(1+m/8)*np.exp2(e-7));return np.where(c&128,-out,out)
w=decode(codes);ordered=[];exact=[];bounds=[]
for q,s in zip(arrays['quant'],arrays['scales']):
 x=decode(q);acc=np.zeros(2,np.float32)
 for j in range(128):acc=(acc.astype(np.float64)+w[:,j]*x[j]).astype(np.float32)
 ordered.append(np.float32(np.float32(acc*s[0])*weight_scale[0]))
 exact.append((w@x)*float(s[0])*float(weight_scale[0]))
 gamma=(260*2**-24)/(1-260*2**-24)
 bounds.append(gamma*(np.abs(w)@np.abs(x))*float(s[0])*float(weight_scale[0])+np.finfo(np.float32).tiny)
arrays.update(weights=weights,weight_scale=weight_scale,ordered=np.array(ordered),exact=np.array(exact),bounds=np.array(bounds))
with Path('fixture.npz').open('xb') as f:np.savez(f,**arrays)
r=dict(fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest(),quant_fixture_sha256=qmeta['fixture_sha256'],quant_groups=len(arrays['groups']),
 model=meta['repository'],revision=meta['revision'],shard_sha256=pin,tensor=name,rows=[0,2],k_block=0,full_model=False,
 criterion='All output FP8 bytes and scale bits exact vs frozen PyTorch; every subtree packet exact; native consumer exact vs independent orderedFP32 emulator and within gamma260 FP64 bound; retained inputs/weight',
 simulation_groups=list(range(12))+[12,20,28,36],physical_groups='all44',scope='Spatial full group128 quantization+packet gathering followed by one original2x128 native FP8 consumer on root; no complete projection/model')
Path('fixture.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
