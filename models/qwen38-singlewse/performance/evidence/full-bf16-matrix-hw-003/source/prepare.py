"""Full original BF16 matrix, representative resident background, independent oracle."""
import hashlib,json
from pathlib import Path
import numpy as np
from weights import OriginalWeights
plan=json.loads(Path('region.json').read_text());meta=json.loads(Path('tensors.json').read_text());background=json.loads(Path('banks.json').read_text())
assert meta['revision']==plan['revision']==background['revision']=='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a'
assert hashlib.sha256(Path('banks.npz').read_bytes()).hexdigest()==background['fixture_sha256']
model=Path('/srv/model-storage/qwen38-singlewse/model');pin={'layers-0.safetensors':'07f700e293baeaf3cd4240c3df1a948c4403f16961ea7979e86c8d6a9f8fd466'}
reader=OriginalWeights(model,meta['tensors'],pin);matrix=reader.rows(plan['matrix']['tensor'],0,48).copy();assert matrix.shape==(48,5120)
bf=lambda c:(c.astype(np.uint32)<<16).view(np.float32)
assert np.isfinite(bf(matrix)).all()
with np.load('banks.npz',allow_pickle=False) as f:old={n:f[n] for n in ['weights','scales','bfweights']}
workers=plan['workers'];bank=np.zeros((960,8814),np.uint32);target=np.zeros((960,128),np.uint32);scale0=np.zeros((960,1),np.uint32)
for w in workers:
 r=w['rank'];x,y=background['owners'][r%6];nx,ny=background['owners'][(r+1)%6];fp=w['fp8_slots'];nb=w['bf16_slots']
 bank[r,:fp*64]=old['weights'][y,x,:fp*128].copy().view(np.uint32)
 bank[r,fp*64:fp*65]=old['scales'][y,x,:fp].view(np.uint32);scale0[r,0]=bank[r,fp*64]
 b=old['bfweights'][y,x].copy()
 if nb==13:b=np.concatenate([b,old['bfweights'][ny,nx,:256]])
 bank[r,fp*65:fp*65+nb*128]=b[:nb*256].view(np.uint32)
 row,k=divmod(r,40);tile=np.ascontiguousarray(matrix[2*row:2*row+2,128*k:128*k+128].T)
 target[r]=tile.view(np.uint32).reshape(-1);begin=fp*65+w['target_slot']*128;bank[r,begin:begin+128]=target[r]
print(json.dumps(dict(phase='complete_original_matrix_packed',matrix_shape=list(matrix.shape),tiles=960,bank_bytes=int(bank.nbytes))),flush=True)
def decode(c):
 c=c.astype(np.int32);e=(c&127)>>3;m=c&7
 return np.where(c&128,-1.,1.)*np.where(e==0,m*2.**-9,(1+m/8)*np.exp2(e-7))
stream=np.zeros((6,40,65),np.uint32);local=np.zeros((6,960,2),np.float32);truth=np.zeros_like(local,dtype=np.float64);absolute=np.zeros_like(truth);cases=[]
ids=np.arange(5120,dtype=np.uint16)
for case in range(5):
 kind=0 if case==4 else 1;mode=['mixed','zero','tiny','boundary','fp8_cohost_smoke'][case]
 if kind==1:
  exponent=(60 if case==2 else 140 if case==3 else 123)+(ids%5)
  codes=((exponent<<7)|((ids*31+13)%128)|((ids%2)<<15)).astype('<u2')
  if case==1:codes=np.where(ids%2,32768,0).astype('<u2')
  stream[case,:,:64]=codes.reshape(40,128).copy().view('<u4');stream[case,:,64]=np.arange(40,dtype=np.uint32)+0x42463136
  inputs=bf(codes).astype(np.float64).reshape(40,128);scales=np.ones(40,np.float32)
 else:
  codes=(((ids*31+13)%127)|((ids%2)*128)).astype(np.uint8);half=(((codes.astype(np.uint16)&127)<<7)|((codes.astype(np.uint16)&128)<<8)).astype('<u2')
  stream[case,:,:64]=half.reshape(40,128).copy().view('<u4');scales=np.exp2(-8-np.arange(40)%3).astype(np.float32);stream[case,:,64]=scales.view(np.uint32);inputs=decode(codes).reshape(40,128)
 for w in workers:
  r=w['rank'];k=w['k'];row=w['group'];fp=w['fp8_slots']
  wd=bf(target[r].view(np.uint16).reshape(128,2).T).astype(np.float64) if kind else decode(bank[r,:64].view(np.uint8).reshape(128,2).T)
  ws=np.float32(1.) if kind else bank[r,fp*64:fp*64+1].view(np.float32)[0];x=inputs[k];acc=np.zeros(2,np.float32)
  for j in range(128):acc=(acc.astype(np.float64)+wd[:,j]*x[j]).astype(np.float32)
  local[case,r]=np.float32(np.float32(acc*scales[k])*ws);factor=float(scales[k])*float(ws)
  truth[case,r]=(wd@x)*factor;absolute[case,r]=(np.abs(wd)@np.abs(x))*factor
 cases.append(dict(case=case,kind='bf16' if kind else 'fp8',mode=mode,complete_original_matrix=bool(kind),slot=6 if kind else 0,replay_of=None))
for a in [stream,local,truth,absolute]:a[5]=a[0]
cases.append(dict(cases[0],case=5,replay_of=0))
trees=np.zeros_like(local);tree_truth=np.zeros_like(truth);tree_bounds=np.zeros_like(truth);counts=np.zeros(960,np.uint32)
for case in range(6):
 for group in range(24):
  def reduce(first,size):
   value=local[case,first].copy();left=size//2;right=size-1-left
   if left:value=np.float32(value+reduce(first+1,left))
   if right:value=np.float32(value+reduce(first+1+left,right))
   trees[case,first]=value;counts[first]=size;tree_truth[case,first]=truth[case,first:first+size].sum(axis=0)
   gamma=((260+size)*2**-24)/(1-(260+size)*2**-24);tree_bounds[case,first]=gamma*absolute[case,first:first+size].sum(axis=0)+np.finfo(np.float32).tiny
   return value
  reduce(group*40,40)
 # Full original matrix mathematical result, independently read row-major above.
 if cases[case]['complete_original_matrix']:
  raw=stream[case,:,:64].copy().view(np.uint16).reshape(5120);wanted=bf(matrix).astype(np.float64)@bf(raw).astype(np.float64)
  np.testing.assert_allclose(tree_truth[case,::40].reshape(48),wanted,rtol=1e-13,atol=1e-300)
gamma=(260*2**-24)/(1-260*2**-24);bounds=gamma*absolute+np.finfo(np.float32).tiny
with Path('fixture.npz').open('xb') as f:np.savez(f,bank=bank,target=target,scale0=scale0,stream=stream,local=local,truth=truth,bounds=bounds,trees=trees,tree_truth=tree_truth,tree_bounds=tree_bounds,counts=counts)
result=dict(model=plan['model'],revision=plan['revision'],matrix=plan['matrix'],workers=workers,application=plan['application'],cases=cases,simulation_cases=[0],physical_cases=list(range(6)),
 full_model=False,full_matrix_shape=[48,5120],fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest(),source_fixture_sha256=background['fixture_sha256'],original_shards=pin,
 background_storage=plan['background_storage'],target_matrix_bytes=int(matrix.nbytes),total_resident_transfer_bytes=int(bank.nbytes),
 simulation_storage='Only complete original BF16 target slot uploaded for BF16 case0; other allocated bytes remain zero. Full original background upload, FP8 smoke case and full bank retention required physically.',
 criterion='All960 native dots and all960 ordered K40 subtrees exactly match frozen FP32 oracle and independent FP64 bounds; all48 original matrix outputs return in order, every streaming root final pair and packet count exact, all input/callback/teardown counters exact, replay and selected/full physical bank retention, normal stop.',scope=plan['scope'])
Path('fixture.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(phase='fixture_ready',sha256=result['fixture_sha256'])),flush=True)
