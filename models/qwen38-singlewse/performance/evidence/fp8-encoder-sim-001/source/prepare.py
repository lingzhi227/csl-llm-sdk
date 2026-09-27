"""Freeze broad finite IEEE inputs and independent PyTorch group-128 FP8 targets."""
import hashlib,json,random,struct
from pathlib import Path
import numpy as np
import torch
from encoder_oracle import bits,LEVELS,oracle

torch.set_num_threads(1)
words={0,0x80000000,1,0x80000001,0x7f7fffff,0xff7fffff}
for level in LEVELS:
 b=bits(level)
 for delta in (-1,0,1):
  if b+delta>=0:words.update([b+delta,(b+delta)|0x80000000])
for left,right in zip(LEVELS,LEVELS[1:]):
 b=bits((left+right)/2)
 for delta in (-1,0,1):words.update([b+delta,(b+delta)|0x80000000])
for exp in range(255):
 for mantissa in (0,1,0x3fffff,0x7ffffe,0x7fffff):
  b=(exp<<23)|mantissa;words.update([b,b|0x80000000])
rng=random.Random(271828)
for _ in range(50000):
 b=rng.getrandbits(32)
 if (b&0x7f800000)!=0x7f800000:words.add(b)
word_array=np.array(sorted(words),np.uint32);values=word_array.view(np.float32)
direct=torch.from_numpy(values.copy()).clamp(-448,448).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
np.testing.assert_array_equal(direct,np.array([oracle(int(b)) for b in word_array],np.uint8))
all_codes=torch.arange(256,dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy();positive=all_codes[:127]
mid=(positive[:-1]+positive[1:])*np.float32(.5)
edges=np.concatenate([mid,np.nextafter(mid,np.float32(-np.inf)),np.nextafter(mid,np.float32(np.inf))])
all_values=np.concatenate([all_codes[np.isfinite(all_codes)],edges,-edges]).astype(np.float32)
groups=np.pad(all_values,(0,(-len(all_values))%128)).reshape(-1,128)
extras=np.stack([np.zeros(128,np.float32),np.sin(np.arange(128)*.31).astype(np.float32),np.linspace(-1e-14,1e-14,128,dtype=np.float32),np.linspace(-448,448,128,dtype=np.float32)])
rng_np=np.random.default_rng(314159)
random_groups=(rng_np.uniform(-1,1,(32,128))*np.exp2(np.arange(32)[:,None]-16)).astype(np.float32)
groups=np.concatenate([groups,extras,random_groups])
x=torch.from_numpy(groups);scales=x.abs().amax(-1,keepdim=True).clamp_min(1e-10)*np.float32(1/448)
quant=(x/scales).clamp(-448,448).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
with Path('fixture.npz').open('xb') as f:np.savez(f,words=word_array,direct=direct,groups=groups,scales=scales.numpy(),quant=quant)
record=dict(fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest(),direct_words=len(words),quant_groups=len(groups),torch_version=torch.__version__,full_model=False,
 scope='All finite FP8 centers and midpoint neighbors, FP32 exponent extremes,50000 seeded random draws;44 group128 vectors with unchanged max/448 and division convention',
 criterion='Exact direct bytes vs independent mathematical and Torch oracle; exact group bytes vs Torch, scale within1ULP; optimized/original device outputs/scales bit-identical; all inputs and untouched output tails retained')
Path('fixture.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
