"""Candidate full-key recurrence oracle with propagated state error; no SDK qualification."""
import math
import struct
N=128
U=2**-24
TINY=2**-126
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def gamma(n):return n*U/(1-n*U)
POLICY={'name':'layer0-recurrent128-four32-candidate-v1','key_dim':128,'value_dim':128,
        'value_columns_per_PE':32,'PEs_per_head':4,
        'boundary':'observed FP32 normalized/scaled q, normalized k, expanded BF16 v/beta and FP32 decay',
        'state_order':'FP32 decay multiply;128 ascending-key FMAs per prediction; subtraction and beta multiply; state update FMA;128 ascending-key output FMAs',
        'reduction':'all128 keys local; four value shards concatenate without arithmetic join',
        'domain':'finite normal FP32 operands and outputs or zero; beta/decay in[0,1]',
        'bounds':'conditional on actual supplied operands; propagate recurrent state rounding across serials',
        'original_arithmetic_engine':'same qualified integer FP32 primitives as Layer3; not a second engine',
        'SDK_qualified':False,'full_model':False}


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


def reduce_full(vector,matrix,matrix_bounds):
    value=[];bound=[]
    for col in range(N):
        terms=[vector[row]*matrix[row*N+col] for row in range(N)]
        perturb=math.fsum(abs(vector[row])*matrix_bounds[row*N+col] for row in range(N))
        magnitude=math.fsum(abs(x) for x in terms)
        value.append(math.fsum(terms))
        bound.append(perturb+gamma(128)*(magnitude+perturb)+128*TINY)
    return value,bound


def step(previous,previous_bounds,token):
    q,k,v,beta,d=(token[n] for n in ('q','k','v','beta','decay'))
    if len(previous)!=N*N or len(previous_bounds)!=N*N or any(len(a)!=N for a in (q,k,v)) or not 0<=beta<=1 or not 0<=d<=1:
        raise ValueError('Invalid recurrence dimensions/domain')
    if any(not math.isfinite(x) for a in (previous,previous_bounds,q,k,v,[beta,d]) for x in a):raise ValueError('Nonfinite input')
    decayed=[d*s for s in previous]
    decayed_bounds=[d*b+U*d*(abs(s)+b)+TINY for s,b in zip(previous,previous_bounds)]
    prediction,pb=reduce_full(k,decayed,decayed_bounds)
    delta=[beta*(a-b) for a,b in zip(v,prediction)]
    db=[beta*b+gamma(2)*beta*(abs(a)+abs(p)+b)+2*TINY for a,p,b in zip(v,prediction,pb)]
    state=[];sb=[]
    for row in range(N):
        for col in range(N):
            i=row*N+col;a=decayed[i];b=decayed_bounds[i];kv=k[row]*delta[col]
            state.append(math.fsum((a,kv)))
            sb.append(b+abs(k[row])*db[col]+U*(abs(a)+b+abs(k[row])*(abs(delta[col])+db[col]))+TINY)
    output,yb=reduce_full(q,state,sb)
    return {'prediction':prediction,'prediction_bounds':pb,'delta':delta,'delta_bounds':db,
            'state':state,'state_bounds':sb,'output':output,'output_bounds':yb}


def check(actual,expected,bounds,exact_zero=False):
    if len(actual)!=len(expected) or len(actual)!=len(bounds) or not actual:raise ValueError('Shape mismatch')
    if any(not math.isfinite(x) for a in (actual,expected,bounds) for x in a) or any(b<=0 for b in bounds):raise ValueError('Invalid gate')
    error=[abs(a-b) for a,b in zip(actual,expected)]
    return {'passed':all(e<=b for e,b in zip(error,bounds)) and (not exact_zero or all(x==0 for x in actual)),
            'max_absolute_error':max(error),'max_error_over_bound':max(e/b for e,b in zip(error,bounds))}
