"""Tiny three-case source recipe; original execution needs separate admission."""
import numpy as np
from .mlp_numerics import Interval, intermediate_product, silu
from .mlp_reference_checks import decode
from .mlp_streamed_source import linear_bounds
from .wp16_reuse_source_admission import require

STAGES=('gate_fp32','gate_bf16','up_fp32','up_bf16','silu_fp32','silu_bf16',
        'product_fp32','product_bf16','down_prefix64_fp32','down_fp32','down_bf16')

def validate_arrays(arrays):
    require(set(arrays)==set(STAGES),'Exactly eleven connected source stages')
    for name,a in arrays.items():
        require(a.dtype==np.dtype('<f8') and a.shape==(3,128,2),'Connected interval shape/dtype')
        require(np.isfinite(a).all() and np.all(a[:,:,0]<=a[:,:,1]),'Finite ordered connected intervals')
    require(sum(a.nbytes for a in arrays.values())==67584,'Exact connected numeric bytes')

def build_intervals(weights,hidden,phase=lambda *args,**kwargs:None):
    require(set(weights)=={'gate','up','down'},'Exact connected weight roles')
    for role,words in weights.items():
        require(words.dtype==np.dtype('<u2') and words.shape==(128,128 if role=='down' else 112),'Selected connected weight shape/dtype')
    require(hidden.dtype==np.dtype('<u2') and hidden.shape==(3,112),'Selected connected input shape/dtype')
    arrays={name:np.empty((3,128,2),dtype='<f8') for name in STAGES}
    inputs=decode(hidden); calls=elements=0
    for role in ('gate','up'):
        for case in range(3):
            points=np.column_stack((inputs[case],inputs[case]))
            for first in (0,64):
                rows=decode(weights[role][first:first+64])
                value=linear_bounds(rows,points); calls+=1; elements+=rows.size
                for suffix in ('fp32','bf16'):arrays[role+'_'+suffix][case,first:first+64]=value[suffix]
                del rows,value
            phase('projection_case',role=role,generation=case+1,input_index=(0,1,3)[case])
    for case in range(3):
        for i in range(128):
            activation=silu(Interval(*map(float,arrays['gate_bf16'][case,i])))
            product=intermediate_product(activation['bf16'],Interval(*map(float,arrays['up_bf16'][case,i])))
            for name,value in (('silu',activation),('product',product)):
                for suffix in ('fp32','bf16'):
                    bounds=value[suffix]; arrays[name+'_'+suffix][case,i]=bounds.lo,bounds.hi
        phase('nonlinear_case',generation=case+1,input_index=(0,1,3)[case])
        for n in (64,128):
            for first in (0,64):
                rows=decode(weights['down'][first:first+64,:n])
                value=linear_bounds(rows,arrays['product_bf16'][case,:n]); calls+=1; elements+=rows.size
                if n==64: arrays['down_prefix64_fp32'][case,first:first+64]=value['fp32']
                else:
                    for suffix in ('fp32','bf16'):arrays['down_'+suffix][case,first:first+64]=value[suffix]
                del rows,value
            phase('down_prefix_case',generation=case+1,input_index=(0,1,3)[case],columns=n)
    require(calls==24 and elements==159744,'Exact connected source contraction recipe')
    validate_arrays(arrays)
    return arrays,dict(linear_bounds_calls=calls,cumulative_weight_elements=elements,scalar_silu_calls=384,scalar_product_calls=384)

def check_anchors(arrays,anchors):
    require(set(anchors)=={'gate','up'},'Two original dense anchors')
    for role,a in anchors.items():
        require(a.dtype==np.dtype('<f8') and a.shape==(128,2),'Dense112 anchor shape/dtype')
        require(arrays[role+'_fp32'][0].tobytes()==a.tobytes(),'Exact accepted dense112 anchor')
    return 512

def sensitivity(arrays):
    validate_arrays(arrays)
    counts={}
    for name in ('gate_bf16','up_bf16','product_bf16','down_bf16'):
        dense,changed=arrays[name][:2]
        counts[name]=int(np.count_nonzero((dense[:,1]<changed[:,0]) | (changed[:,1]<dense[:,0])))
    changed=arrays['down_bf16'][1]
    nonzero=int(np.count_nonzero((changed[:,0]>0) | (changed[:,1]<0)))
    zeros={name:int(np.count_nonzero(np.all(a[2]==0.,axis=1))) for name,a in arrays.items()}
    passed=all(n>0 for n in counts.values()) and nonzero>0 and all(n==128 for n in zeros.values())
    return dict(passed=passed,dense_changed_disjoint_BF16_rows=counts,
        changed_down_intervals_excluding_zero=nonzero,zero_generation_numerical_zero_rows=zeros,
        zero_sign_policy='Numerical source zero; retain raw signed zeros in device casts and retention')
