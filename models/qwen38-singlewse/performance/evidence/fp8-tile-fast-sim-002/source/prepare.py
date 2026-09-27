"""Bounded original layer0 weight slices, independent FP64 oracle and fixed packets."""
import hashlib,json
from pathlib import Path
import numpy as np
from weights import OriginalWeights
model=Path('/srv/model-storage/qwen38-singlewse/model')
if not model.exists():model=Path('/srv/qwen38-singlewse-hardware/model')
meta=json.loads(Path('tensors.json').read_text());assert meta['revision']=='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a'
pin={'layers-0.safetensors':'07f700e293baeaf3cd4240c3df1a948c4403f16961ea7979e86c8d6a9f8fd466'}
reader=OriginalWeights(model,meta['tensors'],pin)

def decode(c):
 c=c.astype(np.int32);e=(c&127)>>3;m=c&7
 v=np.where(e==0,m*2.0**-9,(1+m/8)*np.exp2(e-7));return np.where(c&128,-v,v)

choices=[('mlp.gate_proj',0,0),('mlp.gate_proj',126,39),('mlp.gate_proj',17406,17),
 ('mlp.up_proj',128,3),('mlp.up_proj',1024,31),('mlp.down_proj',5118,135),
 ('linear_attn.in_proj_qkv',10238,39),('linear_attn.out_proj',128,47)]
packed=[];scales=[];packets=[];exact=[];bounds=[];origins=[]
for stem,row,kblock in choices:
 name='model.language_model.layers.0.'+stem+'.weight'
 codes=reader.rows(name,row,2)[:,kblock*128:(kblock+1)*128].copy();assert codes.shape==(2,128) and np.all((codes&127)!=127)
 s=reader.rows(name+'_scale_inv',row//128,1)[0,kblock];scale=(np.array([s],np.uint32)<<16).view(np.float32)[0]
 assert np.isfinite(scale) and scale>0
 weight=np.ascontiguousarray(codes.T).view('<u2').reshape(-1);w=decode(codes)
 for pattern in range(4):
  ids=np.arange(128,dtype=np.int32)
  raw=(np.zeros(128,np.uint8) if pattern==0 else ((ids*(13 if pattern==1 else 31)+7)%127 | ((ids%2)*128)).astype(np.uint8))
  if pattern==3:raw=np.where(ids%4<2,126,254).astype(np.uint8)
  act=np.float32(2.0**(-8 if pattern<2 else -30 if pattern==2 else 0))
  packet=np.zeros(33,np.uint32);packet[:32]=raw.copy().view('<u4');packet[32]=act.view(np.uint32)
  x=decode(raw);e=(w@x)*float(act)*float(scale);sumabs=(np.abs(w)@np.abs(x))*float(act)*float(scale)
  gamma=(260*2**-24)/(1-260*2**-24)
  packed.append(weight);scales.append([scale]*3);packets.append(packet);exact.append(e);bounds.append(gamma*sumabs+np.finfo(np.float32).tiny)
  origins.append(dict(tensor=name,rows=[row,row+2],kblock=kblock,pattern=pattern,activation_scale=float(act),weight_scale=float(scale)))
with Path('fixture.npz').open('xb') as f:np.savez(f,weights=np.array(packed,np.uint16),scales=np.array(scales,np.float32),packets=np.array(packets,np.uint32),exact=np.array(exact),bounds=np.array(bounds))
record=dict(model=meta['repository'],revision=meta['revision'],shard_sha256=pin,cases=len(origins),origins=origins,
 fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest(),criterion='Bit-exact optimized/scalar FP32 outputs (zero signs canonicalized); each within frozen gamma260 absolute-product FP64 bound; all packed16-bit patterns exact; retained inputs',full_model=False)
Path('fixture.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(dict(cases=len(origins),fixture_sha256=record['fixture_sha256'])),flush=True)
