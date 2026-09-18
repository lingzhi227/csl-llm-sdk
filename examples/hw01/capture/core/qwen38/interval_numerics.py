"""Shared finite BF16 interval helpers; exact extracted arithmetic."""
import math
import bisect
from fractions import Fraction
from .rms_numerics import bf16_of_real,bits
from .gated_numerics import expanded

def outward_radius(value,error,magnitude=None):
    if error==0:return 0.0
    magnitude=abs(value) if magnitude is None else magnitude
    exact=Fraction.from_float(error)*(1+Fraction(1,2**40))+abs(Fraction.from_float(magnitude))*Fraction(1,2**44)
    rounded=float(exact)
    return math.nextafter(rounded,math.inf) if Fraction.from_float(rounded)<exact else rounded

def rational_bf16(value):
    if value==0:return 0.0
    center=bits(float(value))>>16
    candidates=[i for i in range(max(0,center-2),min(65535,center+2)+1) if i&0x7f80!=0x7f80]
    chosen=min(candidates,key=lambda i:(abs(Fraction.from_float(expanded(i))-value),i&1))
    return expanded(chosen)

def quantized_interval(value,error):
    if not math.isfinite(value) or not math.isfinite(error) or error<0:raise ValueError('Invalid interval')
    center=expanded(bf16_of_real(value))
    if error==0:return center,0.0
    exact_value=Fraction.from_float(value);exact_error=Fraction.from_float(error)
    lo=rational_bf16(exact_value-exact_error)
    hi=rational_bf16(exact_value+exact_error)
    return center,max(abs(center-lo),abs(center-hi))

def quantized_vector(values,errors):
    if len(values)!=len(errors):raise ValueError('Quantization shape mismatch')
    pairs=[quantized_interval(v,outward_radius(v,e)) for v,e in zip(values,errors)]
    return [p[0] for p in pairs],[p[1] for p in pairs]

def check_interval(actual,expected,bounds):
    if not(len(actual)==len(expected)==len(bounds)) or not actual:raise ValueError('Shape mismatch')
    if any(not math.isfinite(x) for seq in (actual,expected,bounds) for x in seq) or any(x<0 for x in bounds):raise ValueError('Invalid interval gate')
    errors=[abs(a-b) for a,b in zip(actual,expected)]
    ratios=[e/b if b else (0.0 if e==0 else math.inf) for e,b in zip(errors,bounds)]
    return {'passed':all(e<=b for e,b in zip(errors,bounds)), 'max_absolute_error':max(errors),
            'max_error_over_bound':max(ratios),'maximum_bound':max(bounds)}

def output_bf16_spans(values,bounds):
    finite=sorted(set(expanded(i) for i in range(65536) if i&0x7f80!=0x7f80))
    counts=[];lows=[];highs=[]
    for v,e in zip(values,bounds):
        lo=Fraction.from_float(v)-Fraction.from_float(e);hi=Fraction.from_float(v)+Fraction.from_float(e)
        start=bisect.bisect_left(finite,lo);end=bisect.bisect_right(finite,hi)
        counts.append(end-start);lows.append(finite[start]);highs.append(finite[end-1])
    return {'low':lows,'high':highs,'distinct_finite_bf16_counts':counts,
            'max_count':max(counts),'single_value_elements':sum(c==1 for c in counts),
            'max_numeric_span':max(hi-lo for hi,lo in zip(highs,lows))}
