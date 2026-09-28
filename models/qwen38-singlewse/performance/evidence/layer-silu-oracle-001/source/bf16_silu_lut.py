"""Independent exhaustive BF16 -> BF16 SiLU oracle and compact resident table.

Reference is torch.nn.functional.silu(BF16.float()).bfloat16(), including its
finite-domain tail behavior. This file does not reuse the candidate exp/SiLU.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import torch

LOW=0x3b80                 # 2**-8
POSITIVE_STOP=0x4100       # 8; larger positive finite values round to themselves
NEGATIVE_STOP=0x4300       # 128; larger negative magnitudes round to signed zero
POSITIVE_COUNT=POSITIVE_STOP-LOW
TABLE_COUNT=POSITIVE_COUNT+NEGATIVE_STOP-LOW


def build_and_prove():
    torch.set_num_threads(1)
    codes=np.arange(65536,dtype=np.uint16);finite=(codes&0x7f80)!=0x7f80
    selected=codes[finite];values=(selected.astype(np.uint32)<<16).view(np.float32)
    expected=torch.nn.functional.silu(torch.from_numpy(values)).bfloat16().view(torch.uint16).numpy().copy()
    lookup=np.zeros(65536,np.uint16);lookup[selected]=expected
    table=np.concatenate([lookup[LOW:POSITIVE_STOP],lookup[0x8000+LOW:0x8000+NEGATIVE_STOP]])
    assert table.shape==(TABLE_COUNT,) and TABLE_COUNT==3328
    magnitude=selected&0x7fff;sign=selected&0x8000
    actual=np.empty_like(selected)
    tiny=magnitude<LOW
    # Integer ties-to-even halving preserves subnormals on normal-only hardware.
    subnormal=tiny&(magnitude<0x100);normal=tiny&~subnormal
    actual[subnormal]=sign[subnormal]|((magnitude[subnormal]>>1)+((magnitude[subnormal]&3)==3).astype(np.uint16))
    actual[normal]=sign[normal]|(magnitude[normal]-0x80)
    positive=~tiny&(sign==0);negative=~tiny&(sign!=0)
    saturated=positive&(magnitude>=POSITIVE_STOP);actual[saturated]=selected[saturated]
    zero=negative&(magnitude>=NEGATIVE_STOP);actual[zero]=sign[zero]
    in_positive=positive&~saturated;in_negative=negative&~zero
    actual[in_positive]=table[magnitude[in_positive]-LOW]
    actual[in_negative]=table[POSITIVE_COUNT+magnitude[in_negative]-LOW]
    mismatched=np.flatnonzero(actual!=expected)
    if len(mismatched):raise AssertionError([(hex(int(selected[i])),hex(int(actual[i])),hex(int(expected[i]))) for i in mismatched[:16]])
    proof=dict(passed=True,finite_inputs=int(finite.sum()),nonfinite_inputs_excluded=int((~finite).sum()),table_values=TABLE_COUNT,
               table_bytes=int(table.nbytes),table_sha256=hashlib.sha256(table.astype('<u2').tobytes()).hexdigest(),
               torch_version=torch.__version__,oracle='torch.nn.functional.silu(BF16.float()).bfloat16()',
               reference_sha256=hashlib.sha256(expected.astype('<u2').tobytes()).hexdigest(),
               exact_for_every_finite_bf16_input=True,physical=False,neural_model_execution=False,
               scope='Independent CPU proof of compact table/bit-domain rules; no CSL or hardware execution.')
    return table.astype('<u2'),proof


if __name__=='__main__':
    table,proof=build_and_prove()
    with Path('silu-table.npy').open('xb') as f:np.save(f,table)
    with Path('silu-proof.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
    print(json.dumps(proof),flush=True)
