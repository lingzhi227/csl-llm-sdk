"""Frozen forward-error propagation for the reduced WP02 chain."""
import math
import struct
U=2**-24
TINY=2**-126
EPSILON=struct.unpack('<f',struct.pack('<f',1e-6))[0]
POLICY={'name':'two56-sum-rms128-v1','unit_roundoff':U,'epsilon_f32':EPSILON,
        'partial':'gamma56*sumabs + 56*min_normal',
        'sum':'b0+b1+u*(abs(p0)+abs(p1)+b0+b1)+min_normal',
        'squared_sum':'input perturbation plus gamma256 accumulation of upper squared values',
        'sqrt_relative_budget':2**-20,'reciprocal_relative_budget':2**-20,
        'sqrt_policy':'SDK math.sqrt_f32; budget is an independently checked acceptance gate, not an assumed vendor guarantee',
        'normalization':'propagated projection and q interval, sqrt/reciprocal budgets, final FP32 multiply',
        'zero_case':'exact numerical zero at both partials, sum, normalized; signed zero allowed'}


def gamma(n):return n*U/(1-n*U)


def reference(weights,vector):
    if len(weights)!=128*112 or len(vector)!=112:raise ValueError('WP02 shape mismatch')
    partials=[[],[]];pb=[[],[]]
    for pe in range(2):
        for row in range(128):
            terms=[weights[row*112+j]*vector[j] for j in range(pe*56,(pe+1)*56)]
            partials[pe].append(math.fsum(terms));pb[pe].append(gamma(56)*math.fsum(abs(x) for x in terms)+56*TINY)
    sums=[math.fsum([partials[0][r],partials[1][r]]) for r in range(128)]
    sb=[pb[0][r]+pb[1][r]+U*(abs(partials[0][r])+abs(partials[1][r])+pb[0][r]+pb[1][r])+TINY for r in range(128)]
    sq=math.fsum(s*s for s in sums)
    perturb=math.fsum(2*abs(s)*b+b*b for s,b in zip(sums,sb))
    sq_bound=perturb+gamma(256)*math.fsum((abs(s)+b)**2 for s,b in zip(sums,sb))+256*TINY
    q=sq/128+EPSILON
    q_bound=sq_bound/128+U*((sq+sq_bound)/128+EPSILON)+TINY
    lower=max(EPSILON*(1-U),q-q_bound)
    root=math.sqrt(q);low_root=math.sqrt(lower)
    reciprocal_error=(1+POLICY['reciprocal_relative_budget'])/(1-POLICY['sqrt_relative_budget'])-1
    normalized=[s/root for s in sums]
    nb=[b/low_root+abs(s)*q_bound/(low_root*root*(low_root+root))+
        (abs(s)+b)/low_root*(reciprocal_error+U*(1+reciprocal_error))+TINY for s,b in zip(sums,sb)]
    return {'partial':partials,'partial_bound':pb,'summed':sums,'sum_bound':sb,
            'squares':sq,'squares_bound':sq_bound,'q':q,'q_bound':q_bound,
            'normalized':normalized,'normalized_bound':nb}


def check(actual,expected,bounds,exact_zero=False):
    if len(actual)!=len(expected) or len(bounds)!=len(actual) or not actual:raise ValueError('Observation shape mismatch')
    if any(not math.isfinite(x) for seq in (actual,expected,bounds) for x in seq) or any(b<=0 for b in bounds):
        raise ValueError('Nonfinite result or invalid bound')
    errors=[abs(a-e) for a,e in zip(actual,expected)]
    passed=all(e<=b for e,b in zip(errors,bounds)) and (not exact_zero or all(a==0 for a in actual))
    return {'passed':passed,'max_absolute_error':max(errors),'max_error_over_bound':max(e/b for e,b in zip(errors,bounds))}


def validate(events,mode,call):
    if len(events)!=2 or any(len(e)!=14 for e in events):raise ValueError('Event schema mismatch')
    root,sender=events
    if any(type(x) is not int for e in events for x in e):raise ValueError('Event counters must be integer')
    base=all(e[0]==call and e[10]==1 and e[11]==0 for e in events)
    order=(root[2]<root[12]<root[3]) if mode==0 else (root[12]<root[3]<root[2])
    return base and order and min(root[2],root[3])>0 and max(root[2],root[3])<root[4]<root[5]<root[6] and root[8]==root[9]==1 and sender[8]==sender[9]==0 and 0<sender[2]<sender[13]<sender[7]<sender[6]
