"""Original full-K projection rows; independently frozen FP32 schedules and bounds."""
import hashlib,json
from pathlib import Path
import numpy as np
from weights import OriginalWeights
meta=json.loads(Path('tensors.json').read_text());assert meta['revision']=='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a'
plan=json.loads(Path('region.json').read_text());blocks=plan['plan']['blocks'];w,h=max(blocks),len(blocks)
model=Path('/srv/model-storage/qwen38-singlewse/model')
if not model.exists():model=Path('/srv/qwen38-singlewse-hardware/model')
pin={'layers-0.safetensors':'07f700e293baeaf3cd4240c3df1a948c4403f16961ea7979e86c8d6a9f8fd466'}
reader=OriginalWeights(model,meta['tensors'],pin)
stems={40:'mlp.gate_proj',48:'linear_attn.out_proj',136:'mlp.down_proj'}
def decode(c):
 c=c.astype(np.int32);e=(c&127)>>3;m=c&7
 return np.where(c&128,-1.0,1.0)*np.where(e==0,m*2.0**-9,(1+m/8)*np.exp2(e-7))
arrays={n:[] for n in ['weights','scales','raw','wire','local','tree','chain','tree_exact','chain_exact','tree_bounds','chain_bounds','tree_count','chain_count']};origins=[]
for case in range(4):
 weights=np.zeros((h,w,128),np.uint16);scales=np.zeros((h,w,1),np.float32);raw=np.zeros((h,w,33),np.uint32);wire=np.zeros((h,w,65),np.uint32)
 local=np.zeros((h,w,2),np.float32);truth=np.zeros((h,w,2),np.float64);absolute=np.zeros_like(truth)
 for y,n in enumerate(blocks):
  name='model.language_model.layers.0.'+stems[n]+'.weight';shape=meta['tensors'][name]['shape'];assert shape[1]==n*128
  row=[0,126,shape[0]-2,254][case];codes=reader.rows(name,row,2);sb=reader.rows(name+'_scale_inv',row//128,1)[0]
  assert codes.shape==(2,n*128) and np.all((codes&127)!=127)
  origins.append(dict(case=case,row=y,tensor=name,output_row_start=row,output_rows=2,input_columns=n*128,full_input_width=True,full_output_rows=False))
  for k in range(n):
   wc=codes[:,128*k:128*(k+1)].copy();weights[y,k]=np.ascontiguousarray(wc.T).view('<u2').reshape(-1)
   sc=(np.array([sb[k]],np.uint32)<<16).view(np.float32)[0];scales[y,k,0]=sc
   ids=np.arange(128,dtype=np.int32);xc=(((ids*31+k*13+case*7)%127)|(((ids+k)%2)*128)).astype(np.uint8)
   if case==1:xc[:]=0
   if case==3:xc=np.where((ids+k)%4<2,126,254).astype(np.uint8)
   act=np.float32(2.0**(-8-k%3 if case==0 else -30 if case==2 else 0))
   raw[y,k,:32]=xc.copy().view('<u4');raw[y,k,32]=act.view(np.uint32)
   half=(((xc.astype(np.uint16)&127)<<7)|((xc.astype(np.uint16)&128)<<8)).astype('<u2');wire[y,k,:64]=half.view('<u4');wire[y,k,64]=act.view(np.uint32)
   wd,xd=decode(wc),decode(xc);acc=np.zeros(2,np.float32)
   for j in range(128):acc=(acc.astype(np.float64)+wd[:,j]*xd[j]).astype(np.float32)
   local[y,k]=np.float32(np.float32(acc*act)*sc);truth[y,k]=(wd@xd)*float(act)*float(sc);absolute[y,k]=(np.abs(wd)@np.abs(xd))*float(act)*float(sc)
 tree_sum=np.zeros_like(local);chain_sum=np.zeros_like(local);tree_exact=np.zeros_like(truth);chain_exact=np.zeros_like(truth);tree_bounds=np.zeros_like(truth);chain_bounds=np.zeros_like(truth)
 tree_count=np.zeros((h,w),np.uint32);chain_count=np.zeros((h,w),np.uint32)
 for y,n in enumerate(blocks):
  gamma=((260+n)*2**-24)/(1-(260+n)*2**-24)
  # Independent interval recursion; do not read compiler's children/parent table.
  def reduce_interval(first,size):
   value=local[y,first].copy();left_size=(size-1+1)//2;right_size=size-1-left_size
   if left_size:value=np.float32(value+reduce_interval(first+1,left_size))
   if right_size:value=np.float32(value+reduce_interval(first+1+left_size,right_size))
   tree_sum[y,first]=value;tree_count[y,first]=size
   tree_exact[y,first]=truth[y,first:first+size].sum(axis=0);tree_bounds[y,first]=gamma*absolute[y,first:first+size].sum(axis=0)+np.finfo(np.float32).tiny
   return value
  reduce_interval(0,n);value=np.zeros(2,np.float32)
  for k in range(n-1,-1,-1):
   value=local[y,k].copy() if k==n-1 else np.float32(local[y,k]+value)
   chain_sum[y,k]=value;chain_count[y,k]=n-k;chain_exact[y,k]=truth[y,k:n].sum(axis=0);chain_bounds[y,k]=gamma*absolute[y,k:n].sum(axis=0)+np.finfo(np.float32).tiny
  np.testing.assert_array_equal(tree_exact[y,0],truth[y,:n].sum(axis=0))
 for key,val in dict(weights=weights,scales=scales,raw=raw,wire=wire,local=local,tree=tree_sum,chain=chain_sum,tree_exact=tree_exact,chain_exact=chain_exact,tree_bounds=tree_bounds,chain_bounds=chain_bounds,tree_count=tree_count,chain_count=chain_count).items():arrays[key].append(val)
with Path('fixture.npz').open('xb') as f:np.savez(f,**{n:np.array(v) for n,v in arrays.items()})
r=dict(model=meta['repository'],revision=meta['revision'],shard_sha256=pin,application=[w,h],blocks=blocks,cases=4,simulation_cases=[0],physical_cases=[0,1,2,3],origins=origins,fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest(),full_model=False,
 criterion='Every native local dot and every subtree/chain aggregate exactly match separately frozen orderedFP32 emulators (zero signs canonicalized); every aggregate satisfies independent gamma(260+Kblocks) FP64 absolute-product bound; participant counts, packet encoding, input/weight retention and endpoint callbacks exact',
 scope='Full original input widths5120/6144/17408; two output rows per projection. Original FP8 bytes/scales; on-device operand decode and native dots, static binary tree or right-to-left chain. Host supplies already-quantized activation fixture; no dynamic producer/full-M/model.')
Path('fixture.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
