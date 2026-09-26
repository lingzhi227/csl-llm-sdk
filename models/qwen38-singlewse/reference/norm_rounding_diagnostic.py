"""Small diagnostic on saved physical norm outputs; no model execution.

The 65 position-zero operands/gains and both physical norm variants come from
norm-accuracy-hw-001. Only scalar reduction/inverse rounding is investigated;
this does not change the original full-prefix qualification or its failures.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(1);torch.set_num_interop_threads(1)
started=time.monotonic()
with np.load('fixture.npz',allow_pickle=False) as f:
    data={name:f[name] for name in f.files}
assert data['inputs'].shape==data['gain'].shape==data['fp64'].shape==(65,5120)
assert data['actual'].shape==(65,2,5120)

def expand(bits):return (bits.astype(np.uint32)<<16).view(np.float32)
def ordered(bits):
    values=bits.astype(np.int32)
    return np.where(values&32768,-(values&32767),values)
def compare(left,right):
    distance=np.abs(ordered(left)-ordered(right))
    return dict(different=int(np.count_nonzero(distance)),max_bf16_ulp=int(distance.max()))
def hex32(value):return hex(int(np.asarray(value,np.float32).view(np.uint32)))

records=[]
for layer in range(65):
    x=torch.from_numpy(expand(data['inputs'][layer]).copy()).reshape(1,1,5120)
    weight=torch.from_numpy(expand(data['gain'][layer]).copy())
    # The unchanged original Qwen3_5RMSNorm expressions, on these exact inputs.
    variance=x.pow(2).mean(-1,keepdim=True)+1e-6
    inverse=torch.rsqrt(variance)
    output=((x*inverse)*(1.0+weight)).to(torch.bfloat16)
    bits=output.view(torch.uint16).numpy().reshape(-1)
    reciprocal_sqrt=((x*(1.0/torch.sqrt(variance)))*(1.0+weight)).to(torch.bfloat16)
    sqrt_bits=reciprocal_sqrt.view(torch.uint16).numpy().reshape(-1)
    square=np.float32(0);correction=np.float32(0);naive=np.float32(0)
    for value in x.numpy().reshape(-1):
        product=np.float32(value*value);naive=np.float32(naive+product)
        adjusted=np.float32(product-correction);following=np.float32(square+adjusted)
        correction=np.float32(np.float32(following-square)-adjusted);square=following
    row=dict(layer=layer,position=0,baseline_vs_original=compare(data['actual'][layer,0],bits),
        compensated_vs_original=compare(data['actual'][layer,1],bits),
        original_vs_fp64=compare(bits,data['fp64'][layer]),
        compensated_vs_fp64=compare(data['actual'][layer,1],data['fp64'][layer]),
        reciprocal_sqrt_vs_original=compare(sqrt_bits,bits),
        torch_mean_hex=hex32(x.pow(2).mean().item()),
        compensated_mean_hex=hex32(np.float32(square/np.float32(5120))),
        naive_mean_hex=hex32(np.float32(naive/np.float32(5120))),
        torch_inverse_hex=hex32(inverse.item()),
        reciprocal_sqrt_hex=hex32((1.0/torch.sqrt(variance)).item()))
    if layer==2:
        indices=np.flatnonzero(data['actual'][layer,1]!=bits)
        row['different_coordinates']=[dict(index=int(i),physical=int(data['actual'][layer,1,i]),
            original=int(bits[i]),fp64=int(data['fp64'][layer,i])) for i in indices[:64]]
    records.append(row)
result=dict(diagnostic=True,full_model_accepted=False,original_criteria_unchanged=True,
    input_source='norm-accuracy-hw-001: baseline002 actual pre-layer inputs at position0',
    cases=len(records),records=records,seconds=time.monotonic()-started,
    fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest())
Path('COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(cases=len(records),seconds=result['seconds'],layer2=records[2])),flush=True)
