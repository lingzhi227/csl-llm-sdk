"""Independent row-major original FP8 MLP reference with explicit rounding.

No candidate packing/CSL helper is imported. Native32/64-wide FMA order and the
balanced local-left-right sum are explicit; independent FP64 row-major truth
bounds each complete projection before BF16. All arrays are bounded row strips.
"""
import math
import numpy as np


def bf16_float(bits):return (np.asarray(bits,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32)
def bf16_bits(value):
    a=np.asarray(value,dtype=np.float32);assert np.isfinite(a).all()
    bits=a.view(np.uint32);return ((bits+np.uint32(32767)+((bits>>16)&1))>>16).astype(np.uint16)


def fp8_float(codes):
    c=np.asarray(codes,dtype=np.uint8);assert np.all((c&127)!=127)
    e=(c&127)>>3;m=c&7
    return np.where(c&128,-1.,1.)*np.where(e==0,m.astype(np.float64)*2.**-9,(1+m.astype(np.float64)/8)*np.exp2(e.astype(np.int16)-7))


LEVELS=fp8_float(np.arange(127,dtype=np.uint8))

def quantize(bf16):
    value=bf16_float(bf16).reshape(-1,128);assert np.isfinite(value).all()
    scale=np.float32(np.maximum(np.max(np.abs(value),axis=1),np.float32(1e-10))*np.float32(1/448))
    normalized=np.float32(value/scale[:,None]);magnitude=np.abs(normalized).astype(np.float64)
    upper=np.minimum(np.searchsorted(LEVELS,magnitude,side='left'),126);lower=np.maximum(upper-1,0)
    du=np.abs(LEVELS[upper]-magnitude);dl=np.abs(magnitude-LEVELS[lower])
    take_upper=(du<dl)|((du==dl)&((upper&1)==0));codes=np.where(take_upper,upper,lower).astype(np.uint8)
    codes|=(np.signbit(normalized).astype(np.uint8)<<7)
    return codes.reshape(-1),scale


def central_tree(local):
    """Independent balanced inorder definition, no generated routing metadata."""
    def visit(lo,hi):
        middle=(lo+hi)//2;value=local[:,middle].copy()
        if lo<middle:value=np.float32(value+visit(lo,middle))
        if middle+1<hi:value=np.float32(value+visit(middle+1,hi))
        return value
    return visit(0,local.shape[1])


def projection(reader,matrix,operand,*,progress=None):
    m,k=matrix['shape'];rows,columns=matrix['tile_shape'];assert rows*columns==256 and k%128==0
    codes,activation_scales=quantize(operand);x=fp8_float(codes).reshape(k//columns,columns)
    ascale=np.repeat(activation_scales,128//columns)
    result=np.empty(m,np.float32);truth=np.empty(m,np.float64);bounds=np.empty(m,np.float64)
    for group in matrix['group_partitions']:
        first=group['output_start']*rows;stop=first+group['output_count']*rows;workers=group['workers']
        for begin in range(first,stop,128):
            count=min(128,stop-begin);assert count%rows==0
            raw=reader.rows(matrix['tensor'],begin,count);weight=fp8_float(raw).reshape(count,k//columns,columns)
            scale_first=begin//128;scale_stop=(begin+count+127)//128
            scale_raw=reader.rows(matrix['scale_tensor'],scale_first,scale_stop-scale_first)
            scale_rows=(np.arange(begin,begin+count)//128)-scale_first
            wscale=np.repeat(bf16_float(scale_raw)[scale_rows],128//columns,axis=1)
            partial=np.zeros((count,k//columns),np.float32)
            for j in range(columns):partial=(partial.astype(np.float64)+weight[:,:,j]*x[None,:,j]).astype(np.float32)
            partial=np.float32(np.float32(partial*ascale[None,:])*wscale)
            if workers==k//columns:local=partial
            else:
                local=np.zeros((count,workers),np.float32)
                for part in range(group['max_k_parts']):
                    start=part*workers;length=min(workers,k//columns-start)
                    local[:,:length]=np.float32(local[:,:length]+partial[:,start:start+length])
            exact=np.sum(weight*x[None,:,:],axis=2)*ascale[None,:].astype(np.float64)*wscale.astype(np.float64)
            absolute=np.sum(np.abs(weight*x[None,:,:]),axis=2)*ascale[None,:].astype(np.float64)*wscale.astype(np.float64)
            expected=central_tree(local);real=np.sum(exact,axis=1);total=np.sum(absolute,axis=1)
            operations=columns+2+group['max_k_parts']-1+math.ceil(math.log2(workers))+1
            gamma=operations*2.**-24/(1-operations*2.**-24);bound=gamma*total+np.finfo(np.float32).tiny
            if not np.all(np.abs(expected.astype(np.float64)-real)<=bound):raise ValueError('Ordered reference exceeds independent FP64 bound')
            result[begin:begin+count]=expected;truth[begin:begin+count]=real;bounds[begin:begin+count]=bound
        if progress:progress(dict(matrix=matrix['tensor'],through_row=stop,total_rows=m))
    return dict(bf16=bf16_bits(result),fp32=result,fp64=truth,bounds=bounds,operand_codes=codes,operand_scales=activation_scales)


def mlp(reader,stage,input_bits,*,progress=None):
    gate=next(r for r in stage['regions'] if r['role']=='gate_up');down=next(r for r in stage['regions'] if r['role']=='down')
    assert [m['shape'] for m in gate['matrices']]==[[17408,5120],[17408,5120]] and down['matrices'][0]['shape']==[5120,17408]
    gp=projection(reader,gate['matrices'][0],input_bits,progress=progress);up=projection(reader,gate['matrices'][1],input_bits,progress=progress)
    # Primary activation oracle uses PyTorch independently of the device table.
    import torch
    torch.set_num_threads(1)
    activated=torch.nn.functional.silu(torch.from_numpy(bf16_float(gp['bf16']))).bfloat16().view(torch.uint16).numpy().copy()
    mixed=bf16_bits(np.float32(bf16_float(activated)*bf16_float(up['bf16'])))
    dp=projection(reader,down['matrices'][0],mixed,progress=progress)
    # Quantization is independently checked against PyTorch conversion too.
    for operand,p in [(input_bits,gp),(mixed,dp)]:
        value=torch.from_numpy(bf16_float(operand).reshape(-1,128));scales=value.abs().amax(-1).clamp_min(1e-10)*np.float32(1/448)
        q=(value/scales[:,None]).to(torch.float8_e4m3fn).view(torch.uint8).numpy().reshape(-1)
        np.testing.assert_array_equal(q,p['operand_codes']);np.testing.assert_array_equal(scales.numpy().view(np.uint32),p['operand_scales'].view(np.uint32))
    return dict(input=input_bits,gate=gp,up=up,activation=activated,mixed=mixed,down=dp)
