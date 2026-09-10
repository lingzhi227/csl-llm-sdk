"""WP15 original-hidden attention source intervals with uncertain persistent K/V."""
import math

from .trained_qk_rope_numerics import (
    Interval, U, FTZ, APPROX, down, up, fp32, bf16, q, input_interval,
    reference as trained_qk_reference, spans,
)
from .rms_numerics import f32

D=256
CAPACITY=8
EXP_RELATIVE=2**-20
SIGMOID_RELATIVE=2**-18
LIBM_RELATIVE=2**-44
POLICY={
    'name':'original-projected-trained-qk-persistent-attention-v1',
    'cases':['bounded_dyadic','changed_dyadic','last_column_onehot','zero_after_nonzero'],
    'request_generations':[1,1,1,2], 'positions':[0,1,2,0],
    'source':'Unchanged WP13 original-hidden BF16 projection centers/radii, all1024 selected rows; WP14 trained RMS/RoPE interval arithmetic at these newly frozen positions. No device observation sets a source bound.',
    'cache':'Keep prior rotated K intervals/nominals and original projected V intervals/nominals for the current request. Reset only on a new request generation, never on contractionID.',
    'dot':'Outward sum256 of interval endpoint products, plus gamma512*sum(max_absolute_products)+512*2^-126. FP32 to BF16 RNE; FP32 scale1/16 then BF16 RNE.',
    'shift':'s_i-max_j(s_j)=min_j(s_i-s_j), with the self term exactly0. Bound each other difference, take endpoint minima, enclose FP32 rounding, and use known nonpositivity. Refuse any shift enclosure below-24.',
    'exp':'Monotone exp endpoints with2^-44 FP64 relative allowance and qualified2^-20 device relative error on[-24,0]. Structural exp(0)=1 from the unchanged polynomial.',
    'probability':'Positive exp sum with gamma_n and FTZ error, positive reciprocal with2^-20 relative error, FP32 exp*inverse, independent BF16 RNE.',
    'value':'Outward interval products of each BF16 probability and uncertain cached projected V; gamma_(2n) magnitude/FTZ reduction bound, then BF16 RNE.',
    'gate':'Monotone ideal sigmoid over the original projected gate interval, FP64 relative2^-44 and unchanged qualified FP32 sigmoid relative2^-18 covering exp/denominator/division. Original gate interval must prove-abs(g) in[-24,0]. BF16 sigmoid then FP32 product and BF16 output.',
    'correlations':'Unknown shared-input/weight/cache correlations are discarded conservatively. Only the exact self-score difference and structural zero are retained. All binary64 endpoint arithmetic rounds outward.',
    'nominal':'Ideal binary64 arithmetic on source BF16 nominals, BF16 at each model boundary; diagnostic comparison only, never used to tighten intervals.',
    'limits':'Fixed four-case CPU sequence and proposed first two connected tokens of one selected head; no upstream hidden RMS, all heads/GQA/output projection/full layer/model/hardware qualification.',
}


def reduction(terms):
    """Products and sums are FP32; interval arithmetic itself is outward FP64."""
    if not terms:
        raise ValueError('Nonempty reduction required')
    total=sum(terms,Interval.point(0))
    if all(t.zero for t in terms):
        return total
    magnitude=0.
    for term in terms:
        magnitude=up(magnitude+term.magnitude)
    operations=2*len(terms)
    gamma=up(operations*U/(1-operations*U))
    return total.widen(up(up(gamma*magnitude)+operations*FTZ))


def exp_interval(value):
    if value.lo < -24 or value.hi > 0:
        raise ValueError('Original-source exponential argument outside[-24,0]')
    if value.zero:
        return Interval.point(1)
    real=Interval(down(math.exp(value.lo)),up(math.exp(value.hi)))
    real=real*Interval(1-LIBM_RELATIVE,1+LIBM_RELATIVE)
    return real*Interval(1-EXP_RELATIVE,1+EXP_RELATIVE)


def sigmoid_interval(value):
    if value.magnitude>24:
        raise ValueError('Projected gate cannot prove the qualified exp domain')
    # Sigmoid is monotone; the existing2^-18 budget already covers the FP32
    # exp/denominator/division implementation for either sign of the input.
    real=Interval(down(1/(1+math.exp(-value.lo))),up(1/(1+math.exp(-value.hi))))
    real=real*Interval(1-LIBM_RELATIVE,1+LIBM_RELATIVE)
    return real*Interval(1-SIGMOID_RELATIVE,1+SIGMOID_RELATIVE)


def attention(q_bounds,keys,values,gate_bounds,q_nominal,key_nominals,value_nominals,gate_nominal):
    n=len(keys)
    if not 1<=n<=CAPACITY or len(values)!=n or len(key_nominals)!=n or len(value_nominals)!=n:
        raise ValueError('Persistent cache prefix/capacity mismatch')
    if any(len(v)!=D for v in [q_bounds,gate_bounds,q_nominal,gate_nominal,*keys,*values,*key_nominals,*value_nominals]):
        raise ValueError('Full D256 operands required')
    dots=[reduction([a*b for a,b in zip(q_bounds,key)]) for key in keys]
    dot_casts=[bf16(x) for x in dots]
    scales=[fp32(x*Interval.point(1/16)) for x in dot_casts]
    scaled=[bf16(x) for x in scales]
    maximum=Interval(max(x.lo for x in scaled),max(x.hi for x in scaled))
    shifts=[]
    for i,score in enumerate(scaled):
        differences=[score+(-other) for j,other in enumerate(scaled) if j!=i]
        real=Interval(min([0.]+[x.lo for x in differences]),min([0.]+[x.hi for x in differences]))
        rounded=fp32(real)
        shift=Interval(rounded.lo,min(0.,rounded.hi))
        if shift.lo < -24:
            raise ValueError('Coupled max-subtracted score enclosure exceeds exp domain')
        shifts.append(shift)
    exps=[exp_interval(x) for x in shifts]
    total=sum(exps,Interval.point(0))
    gamma=up(n*U/(1-n*U))
    denominator=total.widen(up(up(gamma*total.hi)+n*FTZ))
    inverse=denominator.reciprocal()*Interval(1-APPROX,1+APPROX)
    probability=[fp32(x*inverse) for x in exps]
    probability_casts=[bf16(x) for x in probability]
    av=[reduction([probability_casts[t]*values[t][c] for t in range(n)]) for c in range(D)]
    av_casts=[bf16(x) for x in av]
    gates=[sigmoid_interval(x) for x in gate_bounds]
    gate_casts=[bf16(x) for x in gates]
    product=[fp32(a*b) for a,b in zip(av_casts,gate_casts)]
    output=[bf16(x) for x in product]

    nominal_dot=[q(math.fsum(a*b for a,b in zip(q_nominal,key))) for key in key_nominals]
    nominal_scores=[q(f32(x/16)) for x in nominal_dot]
    nominal_exp=[math.exp(f32(x-max(nominal_scores))) for x in nominal_scores]
    nominal_probability=[q(x/math.fsum(nominal_exp)) for x in nominal_exp]
    nominal_av=[q(math.fsum(nominal_probability[t]*value_nominals[t][c] for t in range(n))) for c in range(D)]
    nominal_gate=[q(1/(1+math.exp(-x))) for x in gate_nominal]
    nominal_output=[q(f32(a*b)) for a,b in zip(nominal_av,nominal_gate)]
    return dict(dot=dots,dot_bf16=dot_casts,scale=scales,scaled_bf16=scaled,maximum=[maximum],
                shift=shifts,exp=exps,denominator=[denominator],inverse=[inverse],probability=probability,
                probability_bf16=probability_casts,attention=av,attention_bf16=av_casts,gate=gates,
                gate_bf16=gate_casts,product=product,output=output,
                nominal=dict(dot_bf16=nominal_dot,scaled_bf16=nominal_scores,probability_bf16=nominal_probability,
                             attention_bf16=nominal_av,gate_bf16=nominal_gate,output=nominal_output),
                source_shift_magnitude_upper=max(-x.lo for x in shifts),output_spans=spans(output))


class SourceCache:
    def __init__(self):
        self.generation=0
        self.keys=[];self.values=[];self.key_nominals=[];self.value_nominals=[]

    def step(self,centers,radii,norm_weights,frequencies,generation,position):
        if len(centers)!=1024 or len(radii)!=1024:
            raise ValueError('Complete selected Q/rawgate/K/V projection source required')
        if generation!=self.generation:
            if generation<=self.generation or position!=0:
                raise ValueError('New request must increase generation and start at0')
            self.generation=generation
            self.keys=[];self.values=[];self.key_nominals=[];self.value_nominals=[]
        if position!=len(self.keys) or position>=CAPACITY:
            raise ValueError('Cache position must advance within the current request')
        selected=centers[:256]+centers[512:768]
        errors=radii[:256]+radii[512:768]
        pre=trained_qk_reference(selected,errors,norm_weights,frequencies,position)
        qk=[h['output'] for h in pre['heads']];qkn=[h['nominal'] for h in pre['heads']]
        val=[input_interval(v,e) for v,e in zip(centers[768:],radii[768:])]
        gate=[input_interval(v,e) for v,e in zip(centers[256:512],radii[256:512])]
        self.keys.append(qk[1]);self.values.append(val)
        self.key_nominals.append(qkn[1]);self.value_nominals.append(centers[768:])
        output=attention(qk[0],self.keys,self.values,gate,qkn[0],self.key_nominals,self.value_nominals,centers[256:512])
        return dict(preprocessing=pre,attention=output,generation=generation,position=position,
                    valid_tokens=len(self.keys),key_intervals=list(self.keys),value_intervals=list(self.values))
