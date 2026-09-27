"""Frozen mixed-bank bytes and independent operand-window/native-dot oracles."""
import hashlib,json
from pathlib import Path
import numpy as np
meta=json.loads(Path('banks.json').read_text());plan=json.loads(Path('region.json').read_text())
assert hashlib.sha256(Path('banks.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
with np.load('banks.npz',allow_pickle=False) as f:banks={n:f[n] for n in ['weights','scales','bfweights']}
owners=meta['owners'];workers=plan['workers'];arrays={};provenance=[]
for d in workers:
 rank=d['rank'];x,y=owners[rank%6];nx,ny=owners[(rank+1)%6];fp=d['fp8_slots'];bf=d['bf16_slots']
 arrays['weights_'+str(rank)]=banks['weights'][y,x,:fp*128].copy().view(np.uint32)
 arrays['scale_'+str(rank)]=banks['scales'][y,x,:fp].copy()
 b=banks['bfweights'][y,x].copy()
 if bf==13:b=np.concatenate([b,banks['bfweights'][ny,nx,:256]])
 arrays['bfweights_'+str(rank)]=b.view(np.uint32)
 provenance.append(dict(worker=[d['x'],d['y']],source_bank=[x,y],fp8_prefix_slots=fp,bf16_prefix_slots=12,
                        extra_bf16_slot=dict(source_bank=[nx,ny],source_slot=0) if bf==13 else None))
def decode(c):
 c=c.astype(np.int32);e=(c&127)>>3;m=c&7
 return np.where(c&128,-1.,1.)*np.where(e==0,m*2.**-9,(1+m/8)*np.exp2(e-7))
def bf(c):return (c.astype(np.uint32)<<16).view(np.float32)
control=np.zeros((10,5,4,4),np.uint16);stream=np.zeros((10,780),np.uint32)
packets=np.zeros((10,4,3,65),np.uint32);local=np.zeros((10,4,3,2),np.float32)
truth=np.zeros_like(local,dtype=np.float64);bounds=np.zeros_like(truth);cases=[]
ids=np.arange(128,dtype=np.uint16)
for case in range(8):
 kind=case%2;period=260 if case<2 else 780;mode=['mixed','zero','tiny','boundary'][case//2]
 pair_ids=[(x-1+case)%2 if period==260 else (x-1+3*(case%2)) for x in range(1,4)]
 inputs=[];scales=[]
 for block in range(period//65):
  packet=np.zeros(65,np.uint32)
  if kind==0:
   codes=(((ids*31+block*13)%127)|(((ids+block)%2)*128)).astype(np.uint8)
   if mode=='zero':codes=np.where(ids%2,128,0).astype(np.uint8)
   if mode=='boundary':codes=np.where((ids+block)%4<2,126,254).astype(np.uint8)
   scale=np.float32(2.**(-30 if mode=='tiny' else -block%3 if mode=='boundary' else -8-block%3))
   half=(((codes.astype(np.uint16)&127)<<7)|((codes.astype(np.uint16)&128)<<8)).astype('<u2')
   packet[:64]=half.view('<u4');packet[64]=scale.view(np.uint32);inputs.append(decode(codes));scales.append(scale)
  else:
   ex=(60 if mode=='tiny' else 140 if mode=='boundary' else 123)+(ids%5)
   codes=((ex<<7)|((ids*31+block*13)%128)|(((ids+block)%2)<<15)).astype('<u2')
   if mode=='zero':codes=np.where(ids%2,32768,0).astype('<u2')
   packet[:64]=codes.view('<u4');packet[64]=0x57494E00+block;inputs.append(bf(codes).astype(np.float64));scales.append(1.)
  stream[case,block*65:(block+1)*65]=packet
 control[case,0,0,0]=period
 for x,pair in enumerate(pair_ids,1):control[case,0,x,:2]=[period,pair*130]
 for d in workers:
  x,y,r=d['x'],d['y'],d['rank'];ordinal=(y+case)%2;block=2*pair_ids[x-1]+ordinal
  slots=d['fp8_slots'] if kind==0 else d['bf16_slots'];slot=0 if case<2 else slots//2 if case<4 else slots-1
  control[case,y,x]=[130,ordinal*65,kind,slot];packet=stream[case,block*65:(block+1)*65];packets[case,y-1,x-1]=packet
  if kind==0:
   wd=decode(arrays['weights_'+str(r)].view(np.uint8).reshape(-1,128,2)[slot].T);ws=arrays['scale_'+str(r)][slot]
  else:wd=bf(arrays['bfweights_'+str(r)].view(np.uint16).reshape(-1,128,2)[slot].T).astype(np.float64);ws=np.float32(1.)
  xd=inputs[block];acc=np.zeros(2,np.float32)
  for j in range(128):acc=(acc.astype(np.float64)+wd[:,j]*xd[j]).astype(np.float32)
  factor=float(scales[block])*float(ws);local[case,y-1,x-1]=np.float32(np.float32(acc*scales[block])*ws)
  truth[case,y-1,x-1]=(wd@xd)*factor
  gamma=(260*2**-24)/(1-260*2**-24);bounds[case,y-1,x-1]=gamma*(np.abs(wd)@np.abs(xd))*factor+np.finfo(np.float32).tiny
 cases.append(dict(case=case,kind=['fp8','bf16'][kind],mode=mode,period_words=period,pair_offsets_words=[p*130 for p in pair_ids],replay_of=None))
for target,source in [(8,0),(9,1)]:
 for a in [control,stream,packets,local,truth,bounds]:a[target]=a[source]
 cases.append(dict(cases[source],case=target,replay_of=source))
arrays.update(control=control,stream=stream,packets=packets,local=local,truth=truth,bounds=bounds)
with Path('fixture.npz').open('xb') as f:np.savez(f,**arrays)
result=dict(application=[4,5],workers=workers,cases=cases,simulation_cases=[0,1,6,7],physical_cases=list(range(10)),
 fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest(),source_fixture_sha256=meta['fixture_sha256'],
 original_bank_source='mixed-bank-sim-002',provenance=provenance,source_origins=meta['origins'],model=meta['model'],revision=meta['revision'],
 full_model=False,scope=plan['scope'],criterion='Exact selected packets/relay pairs, orderedFP32 native dots (zero signs canonicalized), independent FP64 gamma260 bounds, all callbacks/teardowns, full original banks retained and normal stop. Host reset/rearm between independent cases; replay checked physically.')
Path('fixture.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(phase='fixture_ready',sha256=result['fixture_sha256'])),flush=True)
