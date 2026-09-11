"""Selected original-source recipe for the 8-PE resident MLP, with no I/O."""
import numpy as np
from .resident_spatial_numerics import fragment_source,pair_bounds
from .mlp_streamed_source import quantize

def selection(role):
    if role in ('gate','up'):return [(2*r*5120,2*(r*5120+192)) for r in range(128)]
    if role=='down':return [(0,32768)]
    if role=='hidden':return [(2*g*5120,2*(g*5120+192)) for g in (0,1,3)]
    raise ValueError('Four declared source selections')

def expanded(bits):
    if bits.dtype!=np.dtype('<u2'):raise ValueError('Exact little-endian BF16 source bits')
    return (bits.astype('<u4')<<16).view('<f4').astype(np.float64)

def build_intervals(weights,hidden,phase=lambda name:None):
    if set(weights)!={'gate','up','down'} or hidden.shape!=(3,192):raise ValueError('Exact spatial roles/generations')
    for role,w in weights.items():
        if w.shape!=(128,128 if role=='down' else 192):raise ValueError('Exact original selected shape')
    source_weights={role:expanded(w) for role,w in weights.items()};h=expanded(hidden)
    result={}
    for g in range(3):
        phase('source_generation_'+str(g+1))
        source=fragment_source(source_weights,h[g])
        for name,value in source.items():
            if name not in result:result[name]=np.empty((3,128,2),np.float64)
            result[name][g]=value
    if len(result)!=16 or sum(a.nbytes for a in result.values())!=98304:raise ValueError('Exact spatial source array accounting')
    return result

def sensitivity(source):
    def disjoint(a,b):return int(np.count_nonzero((a[:,1]<b[:,0])|(b[:,1]<a[:,0])))
    stages=('gate_bf16','up_bf16','product_bf16','down_bf16')
    changed={name:disjoint(source[name][0],source[name][1]) for name in stages}
    nonzero=int(np.count_nonzero((source['down_bf16'][1,:,0]>0)|(source['down_bf16'][1,:,1]<0)))
    zeros=all(np.all(a[2]==0) for a in source.values())
    faults={}
    for role in ('gate','up','down'):
        a=source[role+'_partial0_fp32'][0];b=source[role+'_partial1_fp32'][0]
        correct=source[role+'_bf16'][0]
        faults[role]=dict(missing_second_shard_disjoint_rows=disjoint(correct,quantize(a)),
            early_partial_BF16_disjoint_rows=disjoint(correct,quantize(pair_bounds(quantize(a),quantize(b)))))
    # These gates detect a stale whole vector and missing projection branches.
    # Do not turn a row count into a claim of all-row fault sensitivity.
    passed=all(n>0 for n in changed.values()) and nonzero>0 and zeros and all(
        faults[role]['missing_second_shard_disjoint_rows']>0 for role in ('gate','up'))
    return dict(passed=passed,dense_changed_disjoint_rows=changed,changed_down_nonzero_rows=nonzero,
        all_zero_source_arrays=zeros,dense_fault_disjoint_rows=faults,
        early_cast_counts_are_reported_not_a_new_tolerance=True,
        claim='Whole-vector stale/zero and missing projection branch sensitivity; row counts explicit')
