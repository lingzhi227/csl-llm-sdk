"""Offline original-weight checks, after device release; no official forward.

Each local96-column conditional enclosure uses the accepted linear_bounds
gamma(192,u32), FP64 summation and result-FTZ policy, vectorized over a bounded
upload rectangle. The existing full source enclosure remains conservative:
each output uses n local additions plus (ceil(n/96)-1) chain additions,
at most2n rounding/FTZ sites assumed by its original gamma(2n,u32) bound.
Those loose source intervals never replace current-operand/tail/cast gates.
For this original four-input profile, observed BF16 products must be exactly
representable normal FP32 or zero, which is checked rather than assumed.
"""
import hashlib,json,math
from pathlib import Path
import numpy as np
from full_capture import S,require,state_check,hash_file
from qwen38.mlp_numerics import Interval,FP32_TINY,U32,U64,gamma,silu,intermediate_product
from qwen38.trained_qk_rope_numerics import fp32,down,up

CASES=('dense_dyadic','changed_dyadic','last_column_onehot','zero_after_nonzero')

def words(value):return (value&65535).astype(np.uint16)
def decode(value):return (np.asarray(value,dtype=np.uint32)<<16).view(np.float32)
def encode(value):
    raw=np.asarray(value,np.float32).view(np.uint32)
    require(np.isfinite(value).all(),'Finite cast input')
    return ((raw+np.uint32(0x7fff)+((raw>>16)&1))>>16).astype(np.uint16)
def equal(a,b,label):require(a.shape==b.shape and np.array_equal(a,b),label)
def inside(values,bounds,label):
    value=values.astype(np.float64).reshape(-1)
    require(bounds.shape==(len(value),2) and np.isfinite(value).all() and np.all(bounds[:,0]<=value) and np.all(value<=bounds[:,1]),label)

def local_bounds(weights,inputs):
    """Point-input version of the accepted96-column bound, no scalar casts."""
    require(weights.shape[:-1]==inputs.shape and weights.shape[-2:]==(96,128),'Packed rectangle/input shape')
    wf=decode(weights);raw=inputs.view(np.uint32);exponent=(raw>>23)&255
    require(not np.any(raw&65535) and np.isfinite(inputs).all(),'BF16-exact actual local input')
    require(not np.any((exponent==0)&((raw&0x7fffffff)!=0)),'Unexpected actual subnormal input requires separate FTZ audit')
    we=(weights>>7)&255
    require(not np.any(we==255) and not np.any((we==0)&((weights&32767)!=0)),'Original normal weights or zero padding')
    terms=wf.astype(np.float64)*inputs.astype(np.float64)[...,None]
    require(not np.any((np.abs(terms)<FP32_TINY)&(terms!=0)) and np.max(np.abs(terms))<=float.fromhex('0x1.fffffep127'),
            'Local BF16 products must be normal exact FP32 or zero for the retained source operation-count proof')
    nominal=terms.sum(axis=-2,dtype=np.float64);np.abs(terms,out=terms)
    magnitude=terms.sum(axis=-2,dtype=np.float64);del terms
    gu=lambda x:np.nextafter(x,np.inf);gd=lambda x:np.nextafter(x,-np.inf)
    magnitude_upper=gu(magnitude/down(1.-gamma(96,U64)))
    e64=gu(gamma(96,U64)*magnitude_upper)
    ftz=up(up(192*FP32_TINY)/down(1.-up(192*U32)))
    e32=gu(gu(gamma(192,U32)*magnitude_upper)+ftz)
    low=gd(gd(nominal-e64)-e32);high=gu(gu(nominal+e64)+e32)
    low[magnitude==0]=0;high[magnitude==0]=0
    return low,high

def nonlinear(data,source):
    active={name:data[name].reshape(2*S.channels,128)[::2].reshape(-1) for name in ('gate','up','product','silu','exponential','sigmoid','activation_fp32','product_fp32')}
    for name in ('gate','up','product','silu'):active[name]=words(active[name])
    g=decode(active['gate']);u=decode(active['up']);a=decode(active['silu']);p=decode(active['product'])
    equal(encode(active['activation_fp32']),active['silu'],'All17408SiLU casts')
    equal(encode(active['product_fp32']),active['product'],'All17408product casts')
    for name,val in [('gate',g),('up',u),('silu',a),('product',p)]:inside(val,source[name+'_bf16'],name+' original source enclosure')
    for name,values in [('silu',active['activation_fp32']),('product',active['product_fp32'])]:inside(values,source[name+'_fp32'],name+' source FP32 enclosure')
    for i,(gv,uv,av) in enumerate(zip(g,u,a)):
        require(abs(gv)<=24 and abs(uv)<=24,'Qualified nonlinear domain')
        exp=math.exp(-abs(float(gv)));error=exp*(2**-20+2**-44)+FP32_TINY
        require(abs(float(active['exponential'][i])-exp)<=math.nextafter(error,math.inf),'Actual gate exponential enclosure')
        sig=exp/(1+exp) if gv<0 else 1/(1+exp)
        require(abs(float(active['sigmoid'][i])-sig)<=math.nextafter(sig*(2**-18+2**-44)+FP32_TINY,math.inf),'Actual gate sigmoid enclosure')
        activation=silu(Interval.point(float(gv)));product=intermediate_product(Interval.point(float(av)),Interval.point(float(uv)))
        for bounds,x,y in [(activation,float(active['activation_fp32'][i]),float(av)),(product,float(active['product_fp32'][i]),float(p[i]))]:
            require(bounds['fp32'].lo<=x<=bounds['fp32'].hi and bounds['bf16'].lo<=y<=bounds['bf16'].hi,'Actual-operand nonlinear enclosure')
        interval=fp32(Interval.point(float(gv))*Interval.point(float(active['sigmoid'][i])))
        require(interval.lo<=active['activation_fp32'][i]<=interval.hi,'Observed gate times observed sigmoid')
    return active

def check_epoch(epoch,data,prepared,receipt,hidden,source,official):
    state_check(data['prepared'],epoch,True);state_check(data['state'],epoch)
    equal(official['hidden'],hidden,'Frozen official input association')
    active=nonlinear(data,source)
    rows=0
    for region,nrows,shards,y0 in [('projection',2*S.channels,S.projection_shards,0),('down',S.outputs,S.down_shards,S.down_y)]:
        inputs=data[region+'_input'].reshape(nrows,shards,96)
        partials=data[region+'_partial'].reshape(nrows,shards,128)
        results=data[region+'_result'].reshape(nrows,shards,128)
        rounded=words(data[region+'_rounded']).reshape(nrows,128)
        partial_bits=partials.view(np.uint32)
        require(not np.any((((partial_bits>>23)&255)==0)&((partial_bits&0x7fffffff)!=0)),'Observed partials do not require input-FTZ ambiguity')
        expected=np.zeros(shards*96,np.float32)
        original=hidden if region=='projection' else active['product'];expected[:len(original)]=decode(original)
        for i in range(nrows):equal(inputs[i].reshape(-1).view(np.uint32),expected.view(np.uint32),'Current original/device product input and complete padded tail')
        for b in receipt['transfers']:
            if (b['role']=='gate_up')!=(region=='projection'):continue
            path=Path(prepared)/'packed'/b['file'];shape=(b['height'],b['width'],96,128)
            weight=np.fromfile(path,dtype='<u2').reshape(shape)
            start=b['y']-y0;end=start+b['height'];low,high=local_bounds(weight,inputs[start:end])
            actual=partials[start:end].astype(np.float64)
            require(np.isfinite(actual).all() and np.all(low<=actual) and np.all(actual<=high),'All current-operand local96 FMA bounds')
            rows+=actual.size
        reduced=partials[:,-1].copy()
        for x in range(shards-2,-1,-1):
            reduced=np.add(reduced,partials[:,x],dtype=np.float32)
            bits=reduced.view(np.uint32)
            require(np.isfinite(reduced).all() and not np.any((((bits>>23)&255)==0)&((bits&0x7fffffff)!=0)),'Chain does not require subnormal-mode ambiguity')
        for x in range(shards):equal(results[:,x].view(np.uint32),reduced.view(np.uint32),'Exact actual-partial FP32 chain and every multicast receiver')
        equal(rounded,encode(reduced),'Observed full reduction final BF16 cast')
        if region=='projection':
            equal(rounded[::2].reshape(-1),active['gate'],'Every gate root to nonlinear handoff')
            equal(rounded[1::2].reshape(-1),active['up'],'Every up root to nonlinear handoff')
            inside(reduced[::2].reshape(-1),source['gate_fp32'],'Full original gate source FP32')
            inside(reduced[1::2].reshape(-1),source['up_fp32'],'Full original up source FP32')
        else:
            equal(rounded.reshape(-1),words(data['output']),'All5120 final collector words')
            inside(reduced.reshape(-1),source['down_fp32'],'Original source final FP32')
            inside(decode(rounded).reshape(-1),source['down_bf16'],'Original source final BF16')
    require(rows==2811904,'All21968PE by128 rows checked')
    if epoch==4:
        for key in ('projection_input','projection_partial','projection_result','down_input','down_partial','down_result'):
            require(not np.any(data[key]),'Zero epoch clears every previous matrix value')
        require(not np.any(decode(words(data['output']))),'Zero final output')
    output=words(data['output']);reference=official['output']
    return dict(epoch=epoch,local_conditional_rows=rows,nonlinear_rows=17408,full_dimension=True,
                original_reference_bf16_mismatches=int(np.count_nonzero(output!=reference)),
                source_and_conditional_enclosures_passed=True,casts_and_handoffs_passed=True)
