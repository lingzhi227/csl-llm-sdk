"""Real second row tile, independent PyTorch quantization and FP64 dot oracle."""
import hashlib,json,struct
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(1)
model=Path('/srv/model-storage/qwen38-singlewse/model')
receipt=json.loads((model/'LAYER0-COMPLETE.json').read_text())
shard=model/'layers-0.safetensors'
pin=next(x for x in receipt['files'] if x['file']==shard.name)
h=hashlib.sha256()
with shard.open('rb') as f:
    for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
assert h.hexdigest()==pin['sha256']
with shard.open('rb') as f:
    size=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(size))
name='model.language_model.layers.0.mlp.gate_proj.weight'
w,s=header[name],header[name+'_scale_inv']
assert w['dtype']=='F8_E4M3' and s['dtype']=='BF16'
matrix=np.memmap(shard,mode='r',offset=8+size+w['data_offsets'][0],dtype='u1',shape=tuple(w['shape']))
scale_matrix=np.memmap(shard,mode='r',offset=8+size+s['data_offsets'][0],dtype='<u2',shape=tuple(s['shape']))
codes=np.array(matrix[272:544,:128],copy=True)
scales=(np.array(scale_matrix[2:5,0],dtype=np.uint32)<<16).view(np.float32)
weights=np.ascontiguousarray(codes.T).view('<u2').reshape(-1)
decoded=torch.from_numpy(codes).view(torch.float8_e4m3fn).float().numpy()
vectors=np.stack([np.sin(np.arange(128)*.17).astype(np.float32),
    ((np.arange(128)%19-9)/16).astype(np.float32),np.zeros(128,np.float32),
    np.linspace(-1e-14,1e-14,128,dtype=np.float32),
    np.linspace(-448,448,128,dtype=np.float32)])
x=torch.from_numpy(vectors)
activation_scales=x.abs().amax(-1,keepdim=True).clamp_min(1e-10)*np.float32(1/448)
quantized=(x/activation_scales).clamp(-448,448).to(torch.float8_e4m3fn)
q=quantized.float().numpy().astype(np.float64)
row_scales=scales[(np.arange(272)+16)//128].astype(np.float64)
exact=(q@decoded.astype(np.float64).T)*activation_scales.numpy().astype(np.float64)*row_scales
products=(np.abs(q)@np.abs(decoded.astype(np.float64).T))*activation_scales.numpy().astype(np.float64)*row_scales
gamma=(260*2**-24)/(1-260*2**-24)
bounds=gamma*products+np.finfo(np.float32).tiny
packets=np.zeros((len(vectors),33),np.uint32)
packets[:,:32]=quantized.view(torch.uint8).numpy().copy().view('<u4').reshape(-1,32)
packets[:,32]=activation_scales.numpy().reshape(-1).view(np.uint32)
with Path('fixture.npz').open('xb') as f:
    np.savez(f,weights=weights,scales=scales,vectors=vectors,packets=packets,exact=exact,bounds=bounds)
meta=dict(repository=receipt['repository'],revision=receipt['revision'],source=pin,tensor=name,
    rows=[272,544],columns=[0,128],row_phase=16,cases=len(vectors),
    fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest(),
    criterion='Exact quantized packet bytes/scales; FP64 block dot oracle; gamma(260)*sum(abs(products)); retention and counters',
    scale_order='FP32(dot*activation_scale)*weight_scale; block partial remains FP32',
    full_model=False,host_neural_computation_during_candidate_run=False)
Path('fixture.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)
