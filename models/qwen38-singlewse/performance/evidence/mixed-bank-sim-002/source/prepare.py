"""Frozen original FP8/BF16 banks, independent ordered FP32/FP64 oracles."""
import hashlib,json
from pathlib import Path
import numpy as np
from weights import OriginalWeights
meta=json.loads(Path('tensors.json').read_text());assert meta['revision']=='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a'
plan=json.loads(Path('region.json').read_text());w=plan['plan']['mesh_width'];n=plan['plan']['blocks'][0];h=n//w;owners=plan['owners'][0]
model=Path('/srv/model-storage/qwen38-singlewse/model')
if not model.exists():model=Path('/srv/qwen38-singlewse-hardware/model')
pin={'layers-0.safetensors':'07f700e293baeaf3cd4240c3df1a948c4403f16961ea7979e86c8d6a9f8fd466','outside.safetensors':'ddff1d6665a2b39f2612fce0ef955e2436724c565bfbcbc127c7ffd078b698ff'}
reader=OriginalWeights(model,meta['tensors'],pin)
def decode(c):
 c=c.astype(np.int32);e=(c&127)>>3;m=c&7
 return np.where(c&128,-1.0,1.0)*np.where(e==0,m*2.0**-9,(1+m/8)*np.exp2(e-7))
def bf(c):return (c.astype(np.uint32)<<16).view(np.float32)
weights=np.zeros((h,w,112,128),np.uint16);scales=np.zeros((h,w,112),np.float32);bfweights=np.zeros((h,w,12,256),np.uint16);origins=[]
fpnames=['model.language_model.layers.0.'+s+'.weight' for s in ['mlp.gate_proj','linear_attn.out_proj','mlp.down_proj']]
bfnames=['lm_head.weight','model.language_model.embed_tokens.weight','model.language_model.layers.0.linear_attn.in_proj_a.weight','model.language_model.layers.0.linear_attn.in_proj_b.weight']
for rank,(x,y) in enumerate(owners):
 for kind,slots,names in [('fp8',112,fpnames),('bf16',12,bfnames)]:
  for slot in range(slots):
   name=names[slot%len(names)];m,k=meta['tensors'][name]['shape'];row=[0,(m//4//2)*2,m-2][(slot//len(names))%3];column=(rank+7*slot)%(k//128)
   values=reader.rows(name,row,2)[:,column*128:(column+1)*128].copy()
   assert np.isfinite(decode(values) if kind=='fp8' else bf(values)).all()
   packed=np.ascontiguousarray(values.T).reshape(-1)
   if kind=='fp8':
    assert np.all((values&127)!=127);weights[y,x,slot]=packed.view('<u2')
    sc=reader.rows(name+'_scale_inv',row//128,1)[0,column];scales[y,x,slot]=bf(np.array([sc],np.uint16))[0]
   else:bfweights[y,x,slot]=packed
   origins.append(dict(pe=[x,y],kind=kind,slot=slot,tensor=name,output_row_start=row,output_rows=2,column_start=column*128,input_columns=128))
print(json.dumps(dict(phase='original_banks_ready',tiles=len(origins))),flush=True)
arrays={key:[] for key in ['control','packet','local','tree','exact','bounds','count']};cases=[]
for case in range(12):
 control=np.zeros((h,w,2),np.uint16);packet=np.zeros((h,w,65),np.uint32);local=np.zeros((h,w,2),np.float32);truth=np.zeros_like(local,dtype=np.float64);absolute=np.zeros_like(truth)
 for rank,(x,y) in enumerate(owners):
  kind=case%2;slot=([0,56,111][case//2] if kind==0 else [0,6,11][case//2]) if case<6 else (rank*13%112 if kind==0 else rank*5%12)
  if case in (8,9):kind=1 if case==8 else 0;slot=0
  if case>=10:kind=(rank+case)%2;slot=(rank*17+case)%(112 if kind==0 else 12)
  control[y,x]=[kind,slot];ids=np.arange(128,dtype=np.uint16);mode='zero' if case in (2,3) else 'tiny' if case in (4,5) else 'boundary' if case in (6,7) else 'mixed'
  if kind==0:
   wc=weights[y,x,slot].view(np.uint8).reshape(128,2).T;wd=decode(wc)
   xc=(((ids*31+rank*13)%127)|(((ids+rank)%2)*128)).astype(np.uint8)
   if mode=='zero':xc=np.where(ids%2,128,0).astype(np.uint8)
   if mode=='boundary':xc=np.where((ids+rank)%4<2,126,254).astype(np.uint8)
   scale=np.float32(2.0**(-30 if mode=='tiny' else 0 if mode=='boundary' else -8-rank%3));xd=decode(xc)
   half=(((xc.astype(np.uint16)&127)<<7)|((xc.astype(np.uint16)&128)<<8)).astype('<u2');packet[y,x,:64]=half.view('<u4');packet[y,x,64]=scale.view(np.uint32)
   acc=np.zeros(2,np.float32)
   for j in range(128):acc=(acc.astype(np.float64)+wd[:,j]*xd[j]).astype(np.float32)
   sc=scales[y,x,slot];local[y,x]=np.float32(np.float32(acc*scale)*sc);factor=float(scale)*float(sc)
  else:
   wc=bfweights[y,x,slot].reshape(128,2).T;wd=bf(wc).astype(np.float64)
   exp=(60 if mode=='tiny' else 140 if mode=='boundary' else 123)+(ids%5)
   xc=((exp<<7)|((ids*31+rank*13)%128)|(((ids+rank)%2)<<15)).astype('<u2')
   if mode=='zero':xc=np.where(ids%2,32768,0).astype('<u2')
   packet[y,x,:64]=xc.view('<u4');packet[y,x,64]=0x4D495845;xd=bf(xc).astype(np.float64)
   acc=np.zeros(2,np.float32)
   for j in range(128):acc=(acc.astype(np.float64)+wd[:,j]*xd[j]).astype(np.float32)
   local[y,x]=acc;factor=1.0
  truth[y,x]=(wd@xd)*factor;absolute[y,x]=(np.abs(wd)@np.abs(xd))*factor
 tree_sum=np.zeros_like(local);exact=np.zeros_like(truth);bounds=np.zeros_like(truth);count=np.zeros((h,w),np.uint32);gamma=((260+n)*2**-24)/(1-(260+n)*2**-24)
 # Independent preorder interval recursion; no compiler children/parent table.
 def reduce(first,size):
  x,y=owners[first];value=local[y,x].copy();ls=size//2;rs=size-1-ls
  if ls:value=np.float32(value+reduce(first+1,ls))
  if rs:value=np.float32(value+reduce(first+1+ls,rs))
  tree_sum[y,x]=value;count[y,x]=size;xy=owners[first:first+size]
  exact[y,x]=sum((truth[yy,xx] for xx,yy in xy),np.zeros(2,np.float64));bounds[y,x]=gamma*sum((absolute[yy,xx] for xx,yy in xy),np.zeros(2,np.float64))+np.finfo(np.float32).tiny
  return value
 reduce(0,n)
 for key,val in dict(control=control,packet=packet,local=local,tree=tree_sum,exact=exact,bounds=bounds,count=count).items():arrays[key].append(val)
 cases.append(dict(case=case,operand_mode=mode,types=[int(control[y,x,0]) for x,y in owners],slots=[int(control[y,x,1]) for x,y in owners]))
for key in arrays:arrays[key]=np.array(arrays[key])
# Replay checks are independent of the device. Both first bank slots recur after switches.
for first,last in [(0,9),(1,8)]:
 for key in arrays:np.testing.assert_array_equal(arrays[key][first],arrays[key][last])
with Path('fixture.npz').open('xb') as f:np.savez(f,weights=weights.reshape(h,w,-1),scales=scales,bfweights=bfweights.reshape(h,w,-1),**arrays)
r=dict(model=meta['repository'],revision=meta['revision'],shard_sha256=pin,application=[w,h],owners=owners,fp8_slots=112,bf16_slots=12,cases=cases,simulation_cases=list(range(12)),physical_cases=list(range(12)),origins=origins,fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest(),full_model=False,
 criterion='Every local dot and subtree aggregate exactly match separately frozen orderedFP32 emulation (zero signs canonicalized); FP64 gamma(260+participants) absolute-product bound; every participant/callback and packet/control exact; all112 FP8 and12 BF16 slots retained after12 epochs without weight reload; repeated first slots exact',
 scope='Six-PE mixed resident bank component; two output rows/tile,128 columns/tile; distinct original slices across slots, not a complete matrix. Host provides typed slot controls and already-encoded operands. BF16 executes finite normal/tiny normal/signed-zero activations with exact high-half expansion; no BF16 subnormal arithmetic qualification, embedding lookup, dynamic producer or full-model schedule.')
Path('fixture.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(phase='fixture_ready',fixture_sha256=r['fixture_sha256'],cases=12)),flush=True)
