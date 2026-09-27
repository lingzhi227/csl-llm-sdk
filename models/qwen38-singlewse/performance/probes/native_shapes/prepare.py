"""Original 256-weight tiles with independent ordered FP32 and FP64 references."""
import hashlib,json
from pathlib import Path
import numpy as np
from weights import OriginalWeights

plan=json.loads(Path('region.json').read_text());meta=json.loads(Path('tensors.json').read_text())
background=json.loads(Path('banks.json').read_text())
assert meta['revision']==background['revision']=='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a'
assert hashlib.sha256(Path('banks.npz').read_bytes()).hexdigest()==background['fixture_sha256']
pin={'layers-0.safetensors':'07f700e293baeaf3cd4240c3df1a948c4403f16961ea7979e86c8d6a9f8fd466'}
reader=OriginalWeights(Path('/srv/model-storage/qwen38-singlewse/model'),meta['tensors'],pin)
fpname='model.language_model.layers.0.mlp.gate_proj.weight'
bfname='model.language_model.layers.0.linear_attn.in_proj_a.weight'
fpraw=reader.rows(fpname,0,16)[:,:128].copy();bfraw=reader.rows(bfname,0,16)[:,:128].copy()
bf=lambda value:(value.astype(np.uint32)<<16).view(np.float32)
scale=bf(reader.rows(fpname+'_scale_inv',0,1)[0,:1])[0]
assert np.isfinite(scale) and scale>0
with np.load('banks.npz',allow_pickle=False) as f:old={n:f[n] for n in ['weights','scales','bfweights']}
n=len(plan['shapes']);assert n in [4,8]
bank=np.zeros((n,8814),np.uint32);origins=[]
for rank,shape in enumerate(plan['shapes']):
 rows,cols=shape['rows'],shape['columns'];owner=rank%4;x,y=background['owners'][owner];nx,ny=background['owners'][(owner+1)%6]
 bank[rank,:110*64]=old['weights'][y,x,:110*128].copy().view(np.uint32)
 bank[rank,110*64:110*65]=old['scales'][y,x,:110].view(np.uint32)
 b=np.concatenate([old['bfweights'][y,x],old['bfweights'][ny,nx,:256]])
 bank[rank,110*65:]=b.view(np.uint32)
 bank[rank,:64]=np.ascontiguousarray(fpraw[:rows,:cols].T).reshape(-1).view(np.uint32)
 bank[rank,110*64]=scale.view(np.uint32)
 begin=110*65+6*128;bank[rank,begin:begin+128]=np.ascontiguousarray(bfraw[:rows,:cols].T).view(np.uint32).reshape(-1)
 origins.append(dict(pe=[rank,0],rows=rows,columns=cols,fp8_tensor=fpname,bf16_tensor=bfname,
                     original_row_start=0,original_column_start=0,fp8_scale_block=[0,0],
                     background_owner=[x,y],extra_bf16_owner=[nx,ny],background_fixture=background['fixture_sha256']))

def decode(c):
 c=c.astype(np.int32);e=(c&127)>>3;m=c&7
 return np.where(c&128,-1.,1.)*np.where(e==0,m*2.**-9,(1+m/8)*np.exp2(e-7))

packets=[];ordered=[];truth=[];bounds=[];cases=[]
for kind in [0,1]:
 for mode in ['mixed','zero','tiny','boundary']:
  packet=np.zeros((n,65),np.uint32);expected=np.zeros((n,16),np.float32);exact=np.zeros((n,16),np.float64);bound=np.zeros_like(exact)
  for rank,shape in enumerate(plan['shapes']):
   rows,cols=shape['rows'],shape['columns'];ids=np.arange(cols,dtype=np.uint16)
   if kind==0:
    codes=(((ids*31+13)%127)|((ids%2)*128)).astype(np.uint8)
    if mode=='zero':codes=np.where(ids%2,128,0).astype(np.uint8)
    if mode=='boundary':codes=np.where(ids%4<2,126,254).astype(np.uint8)
    scale_x=np.float32(2.**(-30 if mode=='tiny' else 0 if mode=='boundary' else -8))
    half=(((codes.astype(np.uint16)&127)<<7)|((codes.astype(np.uint16)&128)<<8)).astype('<u2')
    packet[rank,:cols//2]=half.view('<u4');packet[rank,cols//2]=scale_x.view(np.uint32)
    weights=decode(fpraw[:rows,:cols]);values=decode(codes);scale_w=scale
   else:
    exponent=(60 if mode=='tiny' else 140 if mode=='boundary' else 123)+(ids%5)
    codes=((exponent<<7)|((ids*31+13)%128)|((ids%2)<<15)).astype('<u2')
    if mode=='zero':codes=np.where(ids%2,32768,0).astype('<u2')
    packet[rank,:cols//2]=codes.view('<u4');packet[rank,cols//2]=0x53485045
    weights=bf(bfraw[:rows,:cols]).astype(np.float64);values=bf(codes).astype(np.float64)
    scale_x=scale_w=np.float32(1.)
   assert np.isfinite(weights).all() and np.isfinite(values).all()
   acc=np.zeros(rows,np.float32)
   for k in range(cols):acc=(acc.astype(np.float64)+weights[:,k]*values[k]).astype(np.float32)
   expected[rank,:rows]=np.float32(np.float32(acc*scale_x)*scale_w)
   factor=float(scale_x)*float(scale_w);exact[rank,:rows]=(weights@values)*factor
   gamma=((2*cols+4)*2**-24)/(1-(2*cols+4)*2**-24)
   bound[rank,:rows]=gamma*(np.abs(weights)@np.abs(values))*factor+np.finfo(np.float32).tiny
  packets.append(packet);ordered.append(expected);truth.append(exact);bounds.append(bound)
  cases.append(dict(case=len(cases),kind=kind,mode=mode,decode_modes=[0,1] if kind==0 else [0]))
arrays=dict(bank=bank,packet=np.array(packets),ordered=np.array(ordered),truth=np.array(truth),bounds=np.array(bounds))
if n==8:
 np.testing.assert_array_equal(bank[:4],bank[4:])
 for key in ['packet','ordered','truth','bounds']:np.testing.assert_array_equal(arrays[key][:,:4],arrays[key][:,4:])
with Path('fixture.npz').open('xb') as f:np.savez(f,**arrays)
result=dict(plan,model=meta['repository'],revision=meta['revision'],original_shards=pin,origins=origins,cases=cases,
            fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest(),
            simulation_cases=list(range(8)),physical_cases=list(range(8)),
            comparison='Equal256 MACs per invocation, with different original tile slices and output dimensions. Per-MAC local timing comparison only; not identical complete-matrix workloads or communication costs.',
            criterion='All active rows exactly match separately emulated orderedFP32 values after zero-sign canonicalization and satisfy independentFP64 bounds; unused output tails remain zero, packet and full original resident bank retained, all32 repetitions counted, normal stop.')
Path('fixture.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(phase='fixture_ready',sha256=result['fixture_sha256'])),flush=True)
