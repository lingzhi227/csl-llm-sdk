"""Independent FP64 recurrence and propagated bounds for a two-way key-row split."""
import math
import struct
N=128
U=2**-24
TINY=2**-126
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def gamma(n):return n*U/(1-n*U)
POLICY={'name':'recurrent128-two64-fma-v1','key_dim':128,'value_dim':128,'key_rows_per_pe':64,
        'boundary':'q/k already normalized, q already scaled; v,beta,decay are explicit FP32 inputs',
        'state_order':'FP32 multiply decay in place; local prediction64 ordered FMAs; root FP32 sum; delta subtraction then multiply; state update FMA; local output64 ordered FMAs; root FP32 sum',
        'state_bound':'prior bound*abs(decay) plus FP32 multiply; update adds key*delta propagated error plus one FMA rounding',
        'reduction_bound':'input perturbation plus gamma64*absolute products on each PE, followed by FP32 two-part sum',
        'delta_bound':'beta*prediction bound plus gamma2*abs(beta)*(abs(v)+abs(prediction)+prediction_bound)',
        'domain':'bounded finite synthetic FP32 inputs; beta and decay in[0,1]; no overflow/subnormal correctness claim',
        'reset':'generation1 adopts uploaded initial state, generation2/3 zero entire device state; token counters restart',
        'official_boundary':'normalization disabled; official divides raw q by sqrt128 and exponentiates g; candidate receives actual resulting FP32 q and decay',
        'source_reference_comparison':{'atol':3e-6,'rtol':3e-6,'scope':'separate tiny official-body comparison, not CSL stage tolerances'}}


def initial_state():return [f32(((r*11+c*7)%29-14)/2048) for r in range(N) for c in range(N)]


def raw_tokens():
    result=[]
    for j,(generation,token,beta,decay,zero) in enumerate(((1,1,0.375,0.5,False),(1,2,0.0,0.5,False),(2,1,0.75,1.0,False),(3,1,0.5,1.0,True))):
        q=[((r*(j+3)+j)%17-8)/16 for r in range(N)]
        k=[((r*(j+5)+2*j)%19-9)/16 for r in range(N)]
        for a in (q,k):
            norm=math.sqrt(math.fsum(x*x for x in a)+1e-6)
            a[:]=[f32(x/norm) for x in a]
        v=[f32(((c*7+j*11)%31-15)/32) for c in range(N)]
        if zero:q=[0.0]*N;k=[0.0]*N;v=[0.0]*N
        result.append({'generation':generation,'token':token,'raw_q':q,'k':k,'v':v,'beta':beta,'requested_decay':decay,'zero_case':zero})
    return result


def reduce_two(vector,matrix,matrix_bounds):
    parts=[];bounds=[]
    for pe in range(2):
        value=[];bound=[]
        for col in range(N):
            terms=[vector[row]*matrix[row*N+col] for row in range(pe*64,(pe+1)*64)]
            perturb=math.fsum(abs(vector[row])*matrix_bounds[row*N+col] for row in range(pe*64,(pe+1)*64))
            magnitude=math.fsum(abs(x) for x in terms)
            value.append(math.fsum(terms));bound.append(perturb+gamma(64)*(magnitude+perturb)+64*TINY)
        parts.append(value);bounds.append(bound)
    total=[math.fsum((a,b)) for a,b in zip(*parts)]
    bound=[a+b+U*(abs(x)+abs(y)+a+b)+TINY for x,y,a,b in zip(parts[0],parts[1],bounds[0],bounds[1])]
    return total,bound


def step(previous,previous_bounds,token):
    q,k,v,beta,d=(token[n] for n in ('q','k','v','beta','decay'))
    if len(previous)!=N*N or len(previous_bounds)!=N*N or any(len(a)!=N for a in (q,k,v)) or not 0<=beta<=1 or not 0<=d<=1:
        raise ValueError('Invalid recurrence dimensions/domain')
    if any(not math.isfinite(x) for a in (previous,previous_bounds,q,k,v,[beta,d]) for x in a):raise ValueError('Nonfinite input')
    decayed=[d*s for s in previous]
    decayed_bounds=[d*b+U*d*(abs(s)+b)+TINY for s,b in zip(previous,previous_bounds)]
    prediction,pb=reduce_two(k,decayed,decayed_bounds)
    delta=[beta*(a-b) for a,b in zip(v,prediction)]
    db=[beta*b+gamma(2)*beta*(abs(a)+abs(p)+b)+2*TINY for a,p,b in zip(v,prediction,pb)]
    state=[];sb=[]
    for row in range(N):
        for col in range(N):
            i=row*N+col;a=decayed[i];b=decayed_bounds[i];kv=k[row]*delta[col]
            state.append(math.fsum((a,kv)))
            sb.append(b+abs(k[row])*db[col]+U*(abs(a)+b+abs(k[row])*(abs(delta[col])+db[col]))+TINY)
    output,yb=reduce_two(q,state,sb)
    return {'prediction':prediction,'prediction_bounds':pb,'delta':delta,'delta_bounds':db,
            'state':state,'state_bounds':sb,'output':output,'output_bounds':yb}


def check(actual,expected,bounds,exact_zero=False):
    if len(actual)!=len(expected) or len(actual)!=len(bounds) or not actual:raise ValueError('Shape mismatch')
    if any(not math.isfinite(x) for a in (actual,expected,bounds) for x in a) or any(b<=0 for b in bounds):raise ValueError('Invalid gate')
    error=[abs(a-b) for a,b in zip(actual,expected)]
    return {'passed':all(e<=b for e,b in zip(error,bounds)) and (not exact_zero or all(x==0 for x in actual)),
            'max_absolute_error':max(error),'max_error_over_bound':max(e/b for e,b in zip(error,bounds))}
