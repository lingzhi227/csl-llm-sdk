"""Actual readback checks for the fixed8PE graph; source intervals stay frozen."""
import math
import numpy as np
from .resident_spatial import TILES
from .resident_sdk_sequence import expected_state
from .resident_spatial_numerics import pair_bounds
from .mlp_reference_checks import decode,encode,rounded
from .mlp_streamed_source import linear_bounds
from .mlp_numerics import Interval,FP32_TINY,silu,intermediate_product
from .trained_qk_rope_numerics import fp32

def require(condition,message):
    if not condition:raise ValueError(message)

def f64(values):
    require(values.dtype==np.float32,'Native FP32 observation required')
    words=values.view(np.uint32);exponent=((words>>23)&255).astype(np.int32)
    require(not np.any(exponent==255),'Finite FP32 observation')
    fraction=(words&0x7fffff).astype(np.int32)
    significant=np.where(exponent==0,fraction,fraction+(1<<23))
    result=np.ldexp(significant.astype(np.float64),np.where(exponent==0,-149,exponent-150))
    return np.copysign(result,np.where(words&0x80000000,-1.,1.))

def inside(values,bounds,label):
    require(bounds.shape==(values.size,2) and np.isfinite(bounds).all() and np.all(bounds[:,0]<=bounds[:,1]),'Valid source pairs: '+label)
    require(np.isfinite(values).all() and np.all((bounds[:,0]<=values)&(values<=bounds[:,1])),'Numerical enclosure: '+label)

def key(name,rank):return name+'@'+('all' if rank is None else str(rank))

def check_state(values,g):
    require(values.dtype==np.uint32 and values.shape==(256,),'Eight complete state vectors')
    require(values.tolist()==[x for rank in range(8) for x in expected_state(rank,g)],'Exact current epoch/release/compute/packet counts')

def check_generation(g,observed,weights,hidden,oracle):
    require(g in (1,2,3),'Fixed generation')
    def raw(name,rank):return observed[key(name,rank)]
    def body(name,rank,bits=False):
        value=raw(name,rank)
        if bits:
            require(value.dtype==np.uint32,'Raw native16 containers retained')
            value=(value&65535).astype('<u2')
            require(value[0]==0xa55a and value[-1]==0x5aa5,'BF16 endpoint guards: '+name)
        else:
            require(value.dtype==np.float32,'Native FP32 buffer')
            require(np.array_equal(value[[0,-1]].view(np.uint32),np.array([1234.5,-4321.25],np.float32).view(np.uint32)),'FP32 endpoint guards: '+name)
        return value[1:-1]
    check_state(raw('state',None),g)
    timers=raw('timing',None)&65535;require(timers.shape==(48,),'Eight timestamp pairs')
    cycles=[]
    for row in timers.reshape(8,6):
        begin=sum(int(row[i])<<(16*i) for i in range(3));end=sum(int(row[3+i])<<(16*i) for i in range(3))
        elapsed=(end-begin)%2**48;require(elapsed>0,'Nonzero local stage timestamp interval');cycles.append(elapsed)
    require(np.array_equal(body('request',4,True),hidden[g-1]),'Current original BF16 request')
    require(np.array_equal(raw('broadcast',4).view(np.uint32),hidden[g-1].astype(np.uint32)<<16),'Exact device input expansion')
    gate=body('gate',5,True);upper=body('up',5,True);product=body('product',5,True);activation=body('silu',5,True)
    for name,root,consumer in (('gate',0,gate),('up',2,upper)):
        require(np.array_equal(body('rounded',root,True),consumer),'Exact current producer-to-nonlinear BF16 handoff: '+name)
    require(np.array_equal(body('rounded',6,True),body('output',4,True)),'Exact current final output handoff')
    require(np.array_equal(raw('broadcast',5).view(np.uint32),product.astype(np.uint32)<<16),'Exact device product expansion')
    dots=0;casts=0
    for role,root,n,input_bits in (('gate',0,96,hidden[g-1]),('up',2,96,hidden[g-1]),('down',6,64,product)):
        parts=[]
        for shard in range(2):
            rank=root+shard;columns=slice(shard*n,(shard+1)*n)
            require(np.array_equal(body('input',rank).view(np.uint32),input_bits[columns].astype(np.uint32)<<16),'Current device contraction input/shard')
            part=f64(body('partial',rank));parts.append(part)
            inside(part,oracle[role+f'_partial{shard}_fp32'][g-1],role+' partial source')
            operands=decode(input_bits[columns]);bounds=np.column_stack((operands,operands));w=decode(weights[role][:,columns])
            conditional=np.concatenate([linear_bounds(w[first:first+64],bounds)['fp32'] for first in (0,64)])
            inside(part,conditional,role+' partial actual-operand check');dots+=128
        reduced=body('reduced',root)
        require(np.array_equal(reduced.view(np.uint32),body('reduced',root+1).view(np.uint32)),'Bit-exact pair result multicast')
        actual=f64(reduced);inside(actual,oracle[role+'_sum_fp32'][g-1],role+' pair source')
        conditional=pair_bounds(*[np.column_stack((p,p)) for p in parts]);inside(actual,conditional,role+' actual-partial pair addition')
        bits=body('rounded',root,True)
        require(np.array_equal(bits,encode(rounded(actual))),'Exact observed FP32 sum to BF16 RNE')
        inside(decode(bits),oracle[role+'_bf16'][g-1],role+' BF16 source');casts+=128
    values={'silu_fp32':f64(body('activation_fp32',5)),'silu_bf16':decode(activation),
            'product_fp32':f64(body('product_fp32',5)),'product_bf16':decode(product)}
    for name,value in values.items():inside(value,oracle[name][g-1],name+' source')
    for stage,bits in (('silu',activation),('product',product)):
        require(np.array_equal(bits,encode(rounded(values[stage+'_fp32']))),'Exact observed nonlinear FP32 to BF16 RNE');casts+=128
    gs=decode(gate);us=decode(upper);acts=decode(activation)
    exp_values=f64(body('exponential',5));sigmoid_values=f64(body('sigmoid',5))
    for i,(gv,uv,av) in enumerate(zip(gs,us,acts)):
        require(abs(gv)<=24 and abs(uv)<=24,'Qualified nonlinear operand domain')
        e=math.exp(-abs(float(gv)));ea=e*(2.**-20+2.**-44)+FP32_TINY
        require(abs(exp_values[i]-e)<=math.nextafter(ea,math.inf),'Actual-gate bounded exp allowance')
        s=e/(1+e) if gv<0 else 1/(1+e);sa=s*(2.**-18+2.**-44)+FP32_TINY
        require(abs(sigmoid_values[i]-s)<=math.nextafter(sa,math.inf),'Actual-gate sigmoid allowance')
        activation_source=silu(Interval.point(float(gv)))
        product_source=intermediate_product(Interval.point(float(av)),Interval.point(float(uv)))
        for prefix,enclosure in (('silu',activation_source),('product',product_source)):
            for kind in ('fp32','bf16'):
                v=values[prefix+'_'+kind][i];iv=enclosure[kind]
                require(iv.lo<=v<=iv.hi,'Observed operand nonlinear enclosure')
        multiplied=fp32(Interval.point(float(gv))*Interval.point(float(sigmoid_values[i])))
        require(multiplied.lo<=values['silu_fp32'][i]<=multiplied.hi,'Actual sigmoid times gate FP32')
    if g==3:
        for t in TILES:
            if t.matrix:
                require(not np.any(f64(body('partial',t.rank))) and not np.any(f64(body('reduced',t.rank))),'Zero after nonzero clears every partial/result')
        require(not np.any(decode(body('output',4,True))),'Zero after nonzero output')
    return dict(generation=g,original_input_index=(0,1,3)[g-1],source_and_conditional_rows=dots,
        exact_FP32_to_BF16_casts=casts,local_stage_cycles=cycles,
        epochs_release_and_packet_counts_passed=True,device_handoffs_bit_exact=True,
        source_intervals_unchanged=True,hardware_latency_measured=False)
