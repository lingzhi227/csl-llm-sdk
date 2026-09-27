"""Frozen autonomous-epoch fixtures derived from the verified P10 original bank."""
import hashlib,json
from pathlib import Path
import numpy as np
source=json.loads(Path('bank-source.json').read_text());meta=json.loads(Path('banks.json').read_text())
assert hashlib.sha256(Path('banks.npz').read_bytes()).hexdigest()==source['fixture_sha256']==meta['fixture_sha256']
assert hashlib.sha256(Path('banks.json').read_bytes()).hexdigest()==source['metadata_sha256']
assert meta['revision']=='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a' and meta['application']==[2,3]
with np.load('banks.npz',allow_pickle=False) as f:weights=f['weights'];scales=f['scales'];bfweights=f['bfweights']
weights=weights.reshape(3,2,112,128);bfweights=bfweights.reshape(3,2,12,256);owners=meta['owners'];n=len(owners);assert n==6
def decode(c):
 c=c.astype(np.int32);e=(c&127)>>3;m=c&7
 return np.where(c&128,-1.0,1.0)*np.where(e==0,m*2.0**-9,(1+m/8)*np.exp2(e-7))
def bf(c):return (c.astype(np.uint32)<<16).view(np.float32)
arrays={key:[] for key in ['local','tree','exact','bounds','count']};requests=[];cases=[]
for case in range(12):
 kind=case%2;slot=([0,56,111,37,0,111][case//2] if kind==0 else [0,6,11,5,0,11][case//2])
 if case in (8,9):kind=1 if case==8 else 0;slot=0
 ids=np.arange(128,dtype=np.uint16);mode='zero' if case in (2,3) else 'tiny' if case in (4,5) else 'boundary' if case in (6,7) else 'mixed'
 request=np.zeros(66,np.uint32);request[0]=kind+(case<<8)+(slot<<16)
 if kind==0:
  xc=(((ids*31)%127)|((ids%2)*128)).astype(np.uint8)
  if mode=='zero':xc=np.where(ids%2,128,0).astype(np.uint8)
  if mode=='boundary':xc=np.where(ids%4<2,126,254).astype(np.uint8)
  scale=np.float32(2.0**(-30 if mode=='tiny' else 0 if mode=='boundary' else -8));xd=decode(xc)
  half=(((xc.astype(np.uint16)&127)<<7)|((xc.astype(np.uint16)&128)<<8)).astype('<u2');request[1:65]=half.view('<u4');request[65]=scale.view(np.uint32)
 else:
  exp=(60 if mode=='tiny' else 140 if mode=='boundary' else 123)+(ids%5)
  xc=((exp<<7)|((ids*31)%128)|((ids%2)<<15)).astype('<u2')
  if mode=='zero':xc=np.where(ids%2,32768,0).astype('<u2')
  request[1:65]=xc.view('<u4');request[65]=0x45504F43;xd=bf(xc).astype(np.float64)
 local=np.zeros((3,2,2),np.float32);truth=np.zeros_like(local,dtype=np.float64);absolute=np.zeros_like(truth)
 for x,y in owners:
  if kind==0:wd=decode(weights[y,x,slot].view(np.uint8).reshape(128,2).T)
  else:wd=bf(bfweights[y,x,slot].reshape(128,2).T).astype(np.float64)
  acc=np.zeros(2,np.float32)
  for j in range(128):acc=(acc.astype(np.float64)+wd[:,j]*xd[j]).astype(np.float32)
  if kind==0:sc=scales[y,x,slot];acc=np.float32(np.float32(acc*scale)*sc);factor=float(scale)*float(sc)
  else:factor=1.0
  local[y,x]=acc;truth[y,x]=(wd@xd)*factor;absolute[y,x]=(np.abs(wd)@np.abs(xd))*factor
 tree_sum=np.zeros_like(local);exact=np.zeros_like(truth);bounds=np.zeros_like(truth);count=np.zeros((3,2),np.uint32);gamma=((260+n)*2**-24)/(1-(260+n)*2**-24)
 def reduce(first,size):
  x,y=owners[first];value=local[y,x].copy();ls=size//2;rs=size-1-ls
  if ls:value=np.float32(value+reduce(first+1,ls))
  if rs:value=np.float32(value+reduce(first+1+ls,rs))
  tree_sum[y,x]=value;count[y,x]=size;xy=owners[first:first+size]
  exact[y,x]=sum((truth[yy,xx] for xx,yy in xy),np.zeros(2,np.float64));bounds[y,x]=gamma*sum((absolute[yy,xx] for xx,yy in xy),np.zeros(2,np.float64))+np.finfo(np.float32).tiny
  return value
 reduce(0,n);assert np.all(np.abs(tree_sum.astype(np.float64)-exact)<=bounds)
 for key,val in dict(local=local,tree=tree_sum,exact=exact,bounds=bounds,count=count).items():arrays[key].append(val)
 requests.append(request);cases.append(dict(case=case,kind=kind,slot=slot,operand_mode=mode))
arrays={key:np.array(val) for key,val in arrays.items()};requests=np.array(requests)
expected=np.concatenate([arrays['local'],arrays['tree']],axis=-1).transpose(1,2,0,3).copy().reshape(3,2,48)
for first,last in [(0,9),(1,8)]:
 for key in arrays:np.testing.assert_array_equal(arrays[key][first],arrays[key][last])
with Path('fixture.npz').open('xb') as f:np.savez(f,weights=weights.reshape(3,2,-1),scales=scales,bfweights=bfweights.reshape(3,2,-1),expected=expected,requests=requests,**arrays)
r=dict(model=meta['model'],revision=meta['revision'],shard_sha256=meta['shard_sha256'],bank_source=source,application=[2,4],workers=[2,3],controller=[0,3],owners=owners,cases=cases,runs=[dict(rounds=96,first=0),dict(rounds=96,first=5)],fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest(),full_model=False,
 criterion='Each of192 autonomous epochs: every worker local/subtree result equals preloaded independent orderedFP32 oracle (zero signs canonicalized); every expected subtree passes frozen independentFP64 bound before dispatch. On-device epoch/count checks, final exact counters, full controller root history/FP64 bounds, reset with rotated case order, all resident bank bytes and oracle/request retention.',
 scope='Six original mixed-bank workers plus one controller and one passive PE.66word request multicast, arrival-triggered native dot and fourword tagged tree result; two96-epoch runs without host access inside each loop.12 preloaded independent operand fixtures repeat eight times; control depends on prior completion but neural operands do not depend on prior output. Not fullK, fullM or full-model/token throughput.')
Path('fixture.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(phase='fixture_ready',fixture_sha256=r['fixture_sha256'],epochs=192)),flush=True)
