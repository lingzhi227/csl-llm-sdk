"""Full128 ordered oracle using the existing qualified integer FP32 engine."""
import struct
from .exact_arithmetic import components32,rounded_integer,word32
def bits(x):return struct.unpack('<I',struct.pack('<f',x))[0]
def real(x):return struct.unpack('<f',struct.pack('<I',x))[0]
def f32(x):return real(bits(x))
def fma(a,b,c):
    va,pa=components32(bits(a));vb,pb=components32(bits(b));vc,pc=components32(bits(c))
    terms=[(v,p) for v,p in [(va*vb,pa+pb),(vc,pc)] if v]
    if not terms:return 0.0
    power=min(p for _,p in terms);integer=sum(v<<(p-power) for v,p in terms)
    return real(word32(rounded_integer(integer,power),power))
def reduce(vector,state):
    out=[0.0]*128
    for r in range(128):
        for c in range(128):out[c]=fma(vector[r],state[r*128+c],out[c])
    return out
def replay(state,token):
    decayed=[f32(token['decay']*v) for v in state]
    prediction=reduce(token['k'],decayed)
    delta=[f32(f32(v-p)*token['beta']) for v,p in zip(token['v'],prediction)]
    updated=[fma(token['k'][r],delta[c],decayed[r*128+c]) for r in range(128) for c in range(128)]
    return dict(prediction=prediction,delta=delta,state=updated,output=reduce(token['q'],updated))
