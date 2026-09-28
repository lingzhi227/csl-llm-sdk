"""Independent row-major full-K mixer projection oracle.

No candidate bank, tile descriptor, native decoder or CSL helper is imported.
The ordered result models128 sequential fused products, explicit original FP8
scale rounding and the fixed right-to-left PE reduction. A separate FP64 result
and a priori absolute roundoff bound check the unrounded FP32 result.
"""
import numpy as np
from reference.mlp_oracle import bf16_float,bf16_bits,fp8_float,quantize


def project_rows(raw, inputs, scale_bits=None):
    raw=np.asarray(raw);inputs=np.asarray(inputs,dtype=np.uint16)
    if raw.ndim!=2 or inputs.ndim!=1 or raw.shape[1]!=len(inputs) or not len(inputs) or len(inputs)%128:
        raise ValueError('Complete group128 contraction required')
    rows,columns=raw.shape;parts=columns//128;bf16=scale_bits is None
    if raw.dtype!=(np.dtype('uint16') if bf16 else np.dtype('uint8')):
        raise ValueError('Original weight representation')
    if bf16:
        weights=bf16_float(raw).astype(np.float64).reshape(rows,parts,128)
        values=bf16_float(inputs).astype(np.float64).reshape(parts,128)
        multipliers=np.ones((rows,parts),np.float64)
    else:
        scales=bf16_float(np.asarray(scale_bits,dtype=np.uint16))
        if scales.shape!=(rows,parts) or not np.isfinite(scales).all() or not np.all(scales>0):
            raise ValueError('Original row/group FP8 scales')
        codes,activation_scales=quantize(inputs)
        weights=(fp8_float(raw)/256.).reshape(rows,parts,128)
        values=(fp8_float(codes)/256.).reshape(parts,128)
        multipliers=65536.*activation_scales.astype(np.float64)[None,:]*scales.astype(np.float64)
    if not np.isfinite(weights).all() or not np.isfinite(values).all():
        raise ValueError('Nonfinite original weights or boundary input')
    acc=np.zeros((rows,parts),np.float32)
    for k in range(128):
        acc=(acc.astype(np.float64)+weights[:,:,k]*values[None,:,k]).astype(np.float32)
    if not bf16:
        acc=np.float32(acc*np.float32(65536.))
        acc=np.float32(acc*activation_scales[None,:])
        acc=np.float32(acc*scales)
    ordered=acc[:,-1].copy()
    for k in range(parts-2,-1,-1):
        ordered=np.float32(acc[:,k]+ordered)
    products=weights*values[None,:,:]
    truth=np.sum(np.sum(products,axis=2,dtype=np.float64)*multipliers,axis=1,dtype=np.float64)
    magnitude=np.sum(np.sum(np.abs(products),axis=2,dtype=np.float64)*multipliers,axis=1,dtype=np.float64)
    operations=128+parts+(0 if bf16 else 3)
    unit=2.**-24;gamma=operations*unit/(1-operations*unit)
    bound=(gamma+columns*4*np.finfo(np.float64).eps)*magnitude
    bound+=4*operations*np.finfo(np.float32).tiny
    if not np.isfinite(ordered).all() or np.any(np.abs(ordered.astype(np.float64)-truth)>bound):
        raise ValueError('Ordered FP32 result violates independent FP64 roundoff bound')
    return dict(bf16=bf16_bits(ordered),ordered_fp32=ordered,fp64=truth,bound=bound)
