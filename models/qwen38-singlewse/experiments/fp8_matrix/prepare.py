"""Freeze full original gate matrix fixture and independent FP64 oracle."""
import hashlib,json,struct
from pathlib import Path
import numpy as np
def decode(c):
    c=c.astype(np.uint32);e=(c&127)>>3;m=c&7
    value=np.where(e==0,m*2.0**-9,(1+m/8)*np.exp2(e.astype(np.int32)-7))
    return np.where(c&128,-value,value)

def encode(x):
    # Independent numerical nearest-neighbor implementation, no device LUT.
    levels=decode(np.arange(127,dtype=np.uint8)).astype(np.float32)
    midpoint=(levels[:-1]+levels[1:])*np.float32(.5)
    magnitude=np.abs(x).clip(0,448)
    nearest=np.searchsorted(midpoint,magnitude,side='left')
    candidate=np.minimum(nearest,125)
    tie=(nearest<126)&(magnitude==midpoint[candidate])
    nearest+=tie&(nearest%2==1)
    return (nearest.astype(np.uint8)|(np.signbit(x).astype(np.uint8)<<7))
shard=Path('layers-0.safetensors')
sha='07f700e293baeaf3cd4240c3df1a948c4403f16961ea7979e86c8d6a9f8fd466'
h=hashlib.sha256()
with shard.open('rb') as f:
    for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
assert h.hexdigest()==sha
with shard.open('rb') as f:
    size=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(size))
name='model.language_model.layers.0.mlp.gate_proj.weight'
w,s=header[name],header[name+'_scale_inv']
assert w['dtype']=='F8_E4M3' and w['shape']==[17408,5120] and s['dtype']=='BF16'
matrix=np.memmap(shard,mode='r',offset=8+size+w['data_offsets'][0],dtype='u1',shape=tuple(w['shape']))
scale_matrix=np.memmap(shard,mode='r',offset=8+size+s['data_offsets'][0],dtype='<u2',shape=tuple(s['shape']))
scales=(np.array(scale_matrix,dtype=np.uint32)<<16).view(np.float32)
packed=np.ascontiguousarray(matrix.reshape(64,272,40,128).transpose(0,2,3,1)).view('<u2').reshape(64,40,17408)
tile_scales=np.stack([scales[(r*272)//128:(r*272)//128+3,:].T for r in range(64)])
vectors=np.stack([np.sin(np.arange(5120)*.17).astype(np.float32),
    ((np.arange(5120)%19-9)/16).astype(np.float32),np.zeros(5120,np.float32)])
x=vectors.reshape(3,40,128)
activation_scales=np.maximum(np.max(np.abs(x),axis=-1,keepdims=True),np.float32(1e-10))*np.float32(1/448)
q=encode(x/activation_scales)
packets=np.zeros((3,40,33),np.uint32)
packets[:,:,:32]=q.copy().view('<u4').reshape(3,40,32)
packets[:,:,32]=activation_scales.reshape(3,40).view(np.uint32)
exact=np.zeros((3,17408),np.float64);products=np.zeros_like(exact)
for k in range(40):
    raw=np.array(matrix[:,k*128:(k+1)*128],copy=True);assert np.all((raw&127)!=127)
    decoded=decode(raw)
    operand=decode(q[:,k]);factor=activation_scales[:,k].astype(np.float64)*np.repeat(scales[:,k],128).astype(np.float64)
    exact+=(operand@decoded.T)*factor;products+=(np.abs(operand)@np.abs(decoded.T))*factor
gamma=(300*2**-24)/(1-300*2**-24);bounds=gamma*products+np.finfo(np.float32).tiny
with Path('fixture.npz').open('xb') as f:
    np.savez(f,weights=packed,scales=tile_scales,vectors=vectors,packets=packets,exact=exact,bounds=bounds)
meta=dict(repository='Qwen/Qwen3.8-27B-FP8',revision='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a',tensor=name,
    shard_sha256=sha,matrix_shape=[17408,5120],application_pes=2600,
    fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest(),
    criterion='Exact device quantized packet broadcast; gamma(300)*sum(abs(products)) FP64 oracle; all weight retention; all counters',
    quantization_oracle='Independent NumPy mathematical decoder and nearest-neighbor encoder with ties-to-even',
    full_model=False,weight_uploads='Once, before three calls; complete original gate projection')
Path('fixture.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)
