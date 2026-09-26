"""Original norm weights and frozen FP64 numerical criteria, before device reads."""
import hashlib,json,struct
from pathlib import Path
import numpy as np

def bf16_bits(x):
    bits=np.asarray(x,np.float32).copy().view(np.uint32)
    return ((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype(np.uint16)
def expand(x):return (x.astype(np.uint32)<<16).view(np.float32)

source=Path('/srv/qwen38-singlewse-hardware/fp8-matrix-hw-001/layers-0.safetensors')
pin='07f700e293baeaf3cd4240c3df1a948c4403f16961ea7979e86c8d6a9f8fd466'
digest=hashlib.sha256()
with source.open('rb') as stream:
    for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
assert digest.hexdigest()==pin
name='model.language_model.layers.0.input_layernorm.weight'
with source.open('rb') as stream:
    size=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(size));spec=header[name]
    assert spec['shape']==[5120] and spec['dtype']=='BF16'
    stream.seek(8+size+spec['data_offsets'][0]);gain=np.frombuffer(stream.read(10240),dtype='<u2').copy()
index=np.arange(5120,dtype=np.float64)
inputs=bf16_bits(np.stack([np.sin(index*.17)*8,np.zeros(5120),(-1)**(index%2)*np.exp2((index%25-8)/2)]))
x=expand(inputs).astype(np.float64);g=expand(gain).astype(np.float64)
exact=x/np.sqrt(np.mean(x*x,axis=1,keepdims=True)+1e-6)*(1+g)
gamma=5200*2.0**-24/(1-5200*2.0**-24)
bound=gamma*np.abs(exact)+np.finfo(np.float32).tiny
lo=expand(bf16_bits(exact-bound));hi=expand(bf16_bits(exact+bound))
probes=np.concatenate([np.linspace(-100,88,112),[-100,-92,-91.9,-90,-88,-87.34,-87.33,-80.1,-80,-79.9,-20,-1,0,1,20,88]]).astype(np.float32)
v=probes.astype(np.float64);e=np.exp(v);sigmoid=np.where(v>=0,1/(1+np.exp(-v)),e/(1+e))
truth=np.stack([e,sigmoid,np.logaddexp(0,v),v*sigmoid])
bounds=np.abs(truth)*np.array([2e-6,2e-6,3e-6,5e-6])[:,None]+2*np.finfo(np.float32).tiny
np.savez('fixture.npz',gain=gain,inputs=inputs,norm_lower=lo,norm_upper=hi,probes=probes,truth=truth,bounds=bounds)
meta=dict(scope='Original5120 norm gain; three BF16 vectors;128 fixed nonlinear probes',tensor=name,shard_sha256=pin,
    criterion='RMSNorm FP64 gamma5200 interval followed by BF16 rounding; nonlinear relative bounds2e-6/2e-6/3e-6/5e-6 plus2*minimum FP32 normal',
    normal_fp32_profile=True,negative_silu_tail_test=True,
    fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest(),full_model=False)
Path('fixture.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))
