"""Fixed two-shard numerical enclosures, reusing accepted local kernel policy.

No payload loader or observed-device values enter this source calculation.
The pair adds *FP32 partials*, then casts once to BF16. A sequential full-width
dot product is not used as a bitwise reference for the changed reduction order.
"""
import numpy as np
from .mlp_streamed_source import linear_bounds,quantize,MAX_F32
from .mlp_numerics import Interval,FP32_TINY,silu,intermediate_product
from .trained_qk_rope_numerics import fp32

def pair_bounds(left,right):
    if left.shape!=right.shape or left.ndim!=2 or left.shape[1]!=2:
        raise ValueError('Two equal FP32 partial interval arrays required')
    result=np.empty_like(left,dtype=np.float64)
    for i,(a,b) in enumerate(zip(left,right)):
        def operand(row):
            x=Interval(*map(float,row))
            if x.magnitude>MAX_F32:raise ValueError('FP32 partial overflow')
            # Enclose devices that flush a subnormal partial operand. This is
            # independent of the final addition's result-FTZ allowance.
            edge=FP32_TINY-2.**-149
            if (x.lo<=edge and x.hi>=2.**-149) or (x.lo<=-2.**-149 and x.hi>=-edge):
                x=Interval(min(0.,x.lo),max(0.,x.hi))
            return x
        value=fp32(operand(a)+operand(b))
        if value.magnitude>MAX_F32:raise ValueError('FP32 pair addition overflow')
        result[i]=value.lo,value.hi
    return result

def projection(weights,inputs,columns):
    if weights.shape!=(128,2*columns) or inputs.shape!=(2*columns,2) or columns not in (64,96):
        raise ValueError('Exact resident fragment shape')
    parts=[]
    for shard in range(2):
        c=slice(shard*columns,(shard+1)*columns)
        parts.append(np.concatenate([linear_bounds(weights[r:r+64,c],inputs[c])['fp32'] for r in (0,64)]))
    summed=pair_bounds(*parts)
    return dict(partial0_fp32=parts[0],partial1_fp32=parts[1],sum_fp32=summed,bf16=quantize(summed))

def fragment_source(weights,hidden):
    if set(weights)!={'gate','up','down'} or hidden.shape!=(192,):raise ValueError('Spatial source roles/input')
    inputs=np.column_stack((hidden,hidden));result={}
    for role in ('gate','up'):
        for key,value in projection(weights[role],inputs,96).items():result[role+'_'+key]=value
    if max(np.abs(result['gate_bf16']).max(),np.abs(result['up_bf16']).max())>24:
        raise ValueError('Resident nonlinear kernel requires both operands within +/-24')
    for name in ('silu_fp32','silu_bf16','product_fp32','product_bf16'):result[name]=np.empty((128,2),np.float64)
    for i in range(128):
        activation=silu(Interval(*map(float,result['gate_bf16'][i])))
        product=intermediate_product(activation['bf16'],Interval(*map(float,result['up_bf16'][i])))
        for name,values in (('silu',activation),('product',product)):
            for kind in ('fp32','bf16'):
                interval=values[kind];result[name+'_'+kind][i]=interval.lo,interval.hi
    for key,value in projection(weights['down'],result['product_bf16'],64).items():result['down_'+key]=value
    return result
