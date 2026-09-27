"""Independent PyTorch oracle for fused BF16/FP8 interfaces; no model claim."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
rng=np.random.default_rng(3827)
arrays={k:[] for k in ['gate','up','down','original','gain','expected']}
def bits(t):return t.contiguous().view(torch.uint16).numpy().astype('<u2')
def words(t):return bits(t).view('<u4').copy()
def quant(t):
    x=t.float();scale=x.abs().max().clamp(min=1e-10)/448.0
    q=(x/scale).to(torch.float8_e4m3fn).view(torch.uint8).numpy().astype(np.uint16)
    half=((q&127)<<7)|((q&128)<<8)
    return np.concatenate([half.astype('<u2').view('<u4'),np.array([scale.item()],'<f4').view('<u4')])
for case in range(3):
    gate=rng.uniform(-12,12,128).astype('<f4');up=rng.uniform(-3,3,128).astype('<f4')
    if case==1:gate[:]=0;up[:]=0
    if case==2:
        gate[:32]=np.tile(np.array([-80,-12,-4,-1,-0.5,0.5,4,12],'<f4'),4)
        # Halfway values with alternating BF16 parity, on both signs.
        gate[32:64]=(np.arange(32,dtype=np.uint32)*0x10000+0x3f008000).view(np.float32)
        gate[64:96]=-gate[32:64]
    down=rng.uniform(-4,4,128).astype('<f4')
    original=torch.from_numpy(rng.uniform(-2,2,128).astype('<f4')).bfloat16()
    gain=torch.from_numpy(rng.uniform(-0.25,0.25,128).astype('<f4')).bfloat16()
    g=torch.from_numpy(gate).bfloat16();u=torch.from_numpy(up).bfloat16()
    activated=torch.nn.functional.silu(g.float()).bfloat16()
    multiplied=(activated.float()*u.float()).bfloat16()
    normalized=((original.float()*torch.tensor(0.75,dtype=torch.float32))*(1.0+gain.float())).bfloat16()
    residual=(torch.from_numpy(down).bfloat16().float()+original.float()).bfloat16()
    expected=np.concatenate([words(g),words(u),words(multiplied),quant(multiplied),words(normalized),quant(normalized),words(residual)]).astype('<u4')
    assert len(expected)==450
    for name,value in dict(gate=gate,up=up,down=down,original=words(original),gain=words(gain),expected=expected).items():arrays[name].append(value)
with Path('fixture.npz').open('xb') as f:np.savez(f,**{k:np.array(v) for k,v in arrays.items()})
metadata=dict(fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest(),cases=3,values_per_case=128,
              criterion='All450 output words per case bit-exact against independent PyTorch BF16/SiLU/group128 FP8 oracle; normal stop; two nonzero epochs separated by zero epoch.',
              torch_version=torch.__version__,physical=False,full_model=False,
              scope='Numerical qualification of fused packet epilogues and local epoch guard only. Synthetic finite operands; no original matrix contraction, distributed readiness, inter-PE stream, throughput or full MLP claim.')
Path('fixture.json').write_text(json.dumps(metadata,indent=2)+'\n')
