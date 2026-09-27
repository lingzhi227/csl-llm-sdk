"""Pinned original projection submatrices and pre-observation numerical bounds."""
import hashlib,json
from pathlib import Path
import numpy as np
from weights import OriginalWeights
model=Path('/srv/model-storage/qwen38-singlewse/model')
if not model.exists():model=Path('/srv/qwen38-singlewse-hardware/model')
plan=json.loads(Path('region.json').read_text())['plan'];w,h=plan['width'],plan['height']
meta=json.loads(Path('tensors.json').read_text());assert meta['revision']=='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a'
pin={'layers-0.safetensors':'07f700e293baeaf3cd4240c3df1a948c4403f16961ea7979e86c8d6a9f8fd466'}
reader=OriginalWeights(model,meta['tensors'],pin)
def decode(c):
 c=c.astype(np.int32);e=(c&127)>>3;m=c&7
 v=np.where(e==0,m*2.0**-9,(1+m/8)*np.exp2(e-7));return np.where(c&128,-v,v)
choices=[('mlp.gate_proj',0,0),('mlp.gate_proj',17408-2*h,40-w),('mlp.up_proj',128,16),('mlp.down_proj',5120-2*h,136-w)]
arrays={n:[] for n in ['weights','scales','raw','wire','ordered','exact','bounds']};origins=[]
for case,(stem,row_start,k_start) in enumerate(choices):
 name='model.language_model.layers.0.'+stem+'.weight'
 weights=np.zeros((h,w,128),np.uint16);scales=np.zeros((h,w,1),np.float32);raw=np.zeros((h,w,33),np.uint32);wire=np.zeros((h,w,65),np.uint32)
 ordered=np.zeros((h,w,2),np.float32);exact=np.zeros((h,w,2),np.float64);bounds=np.zeros_like(exact)
 for y in range(h):
  row=row_start+2*y;codes=reader.rows(name,row,2);sb=reader.rows(name+'_scale_inv',row//128,1)[0]
  accum=np.zeros(2,np.float32);truth=np.zeros(2,np.float64);absolute=np.zeros(2,np.float64)
  for x in range(w):
   k=k_start+x;wc=codes[:,k*128:(k+1)*128].copy();assert np.all((wc&127)!=127)
   weights[y,x]=np.ascontiguousarray(wc.T).view('<u2').reshape(-1)
   scale=(np.array([sb[k]],np.uint32)<<16).view(np.float32)[0];assert np.isfinite(scale) and scale>0
   scales[y,x,0]=scale
   ids=np.arange(128,dtype=np.int32)
   xc=(((ids*(13 if case==1 else 31)+x*7)%127)|((ids%2)*128)).astype(np.uint8)
   if case==0:xc[:]=0
   if case==3:xc=np.where((ids+x)%4<2,126,254).astype(np.uint8)
   act=np.float32(2.0**(-30 if case==2 else 0 if case==3 else -8-x%3))
   raw[y,x,:32]=xc.copy().view('<u4');raw[y,x,32]=act.view(np.uint32)
   half=(((xc.astype(np.uint16)&127)<<7)|((xc.astype(np.uint16)&128)<<8)).astype('<u2')
   wire[y,x,:64]=half.view('<u4');wire[y,x,64]=act.view(np.uint32)
   wd=decode(wc);xd=decode(xc);local=np.zeros(2,np.float32)
   for j in range(128):local=(local.astype(np.float64)+wd[:,j]*xd[j]).astype(np.float32)
   local=np.float32(np.float32(local*act)*scale)
   accum=local if x==0 else np.float32(accum+local)
   truth+=(wd@xd)*float(act)*float(scale);absolute+=(np.abs(wd)@np.abs(xd))*float(act)*float(scale)
   ordered[y,x]=accum;exact[y,x]=truth
   gamma=((260+w)*2**-24)/(1-(260+w)*2**-24)
   bounds[y,x]=gamma*absolute+np.finfo(np.float32).tiny
 for n,v in [('weights',weights),('scales',scales),('raw',raw),('wire',wire),('ordered',ordered),('exact',exact),('bounds',bounds)]:arrays[n].append(v)
 origins.append(dict(tensor=name,row_start=row_start,rows=2*h,k_block_start=k_start,k_blocks=w,pattern=case))
with Path('fixture.npz').open('xb') as f:np.savez(f,**{n:np.array(v) for n,v in arrays.items()})
record=dict(model=meta['repository'],revision=meta['revision'],shard_sha256=pin,application=[w,h],cases=len(choices),origins=origins,fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest(),
 criterion='Every K-prefix bit-exact to independent ordered FP32 emulator (zero signs canonicalized), within gamma(260+Kblocks) FP64 absolute-product bound; operand coding and all endpoint counters/retention exact',full_model=False)
Path('fixture.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
