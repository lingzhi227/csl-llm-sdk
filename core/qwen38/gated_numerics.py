"""Gated RMS128: independent math plus exact conversion of observed FP32 bits."""
import math
from .rms_numerics import f32,bits,from_bits,bf16_rne,bf16_of_real,gamma,U,TINY,EPSILON,close,PROBE_BITS
N=128
POLICY={'name':'gated-rms128-bf16-v1','epsilon':EPSILON,'size':N,
 'domain':'finite BF16 x/gain, abs(x)<=2, abs(gain)<=2, BF16 gate z in[-16,16]; no overflow/subnormal correctness claim',
 'sequence':'BF16 input -> FP32 RMS -> BF16 RNE normalized -> direct BF16 gain product with BF16 RNE -> FP32 SiLU(z.float()) product -> BF16 RNE',
 'composition_input':'recurrent FP32 y is device RNE-cast to BF16; persistent state stays FP32',
 'sqrt_relative':2**-20,'inverse_relative':2**-20,'exp_relative':2**-20,'silu_relative':2**-18,
 'norm_bound':'gamma256 sum plus denominator rounding propagated through sqrt/inverse approximation and multiply',
 'conversion':'exact observed-FP32 RNE at input, normalized, gain product and final output boundaries; mathematical BF16 parity non-gating',
 'gate':'exp(-abs(z)); sigmoid=e/(1+e) for negative z, otherwise1/(1+e); SiLU=z*sigmoid',
 'stage_check':'gain/final product checked against observed preceding BF16/FP32 operands, independently of upstream error',
 'official_comparison':'independent mathematical output within 2^-6*abs(expected)+1e-7; BF16 parity reported separately, not a CSL conversion gate'}


def expanded(raw):return from_bits(raw<<16)
def quantize(value):return expanded(bf16_rne(bits(f32(value))))
def silu(z):
    e=math.exp(-abs(z));return z*(e/(1+e) if z<0 else 1/(1+e))


def fixtures():
    gates=[-16.,-8.,-4.,-2.,-1.,-0.5,0.,0.5,1.,2.,4.,8.,16.]
    result=[]
    for j,name in enumerate(('zero','direct_gain_zero','signed_gate','early_cast_adversary')):
        x=[((i*(3+j*4))%127-63)/32 for i in range(N)]
        w=[((i*3+j)%17-8)/8 for i in range(N)]
        z=[gates[(i+j)%len(gates)] for i in range(N)]
        if j==0:x=[0.0]*N
        if j==1:w=[0.0 if i%3==0 else 1.0 for i in range(N)]
        result.append({'name':name,'x':x,'w':w,'z':z})
    return result


def reference(x,w,z):
    if any(len(a)!=N for a in (x,w,z)):raise ValueError('Expected128 values')
    if any(not math.isfinite(v) or abs(v)>limit or bits(v)&65535 for a,limit in ((x,2),(w,2),(z,16)) for v in a):raise ValueError('Outside BF16 domain')
    squares=math.fsum(a*a for a in x);sb=gamma(256)*squares+256*TINY
    q=squares/N+EPSILON;qb=sb/N+gamma(2)*(abs(squares/N)+EPSILON+sb/N)+2*TINY
    low=max(EPSILON*(1-U),q-qb);rt=math.sqrt(q);lo=math.sqrt(low)
    approx=(1+POLICY['inverse_relative'])/(1-POLICY['sqrt_relative'])-1
    ib=qb/(lo*rt*(lo+rt))+approx/lo
    norm=[a/rt for a in x];nb=[abs(a)*ib+U*(abs(a)/rt+abs(a)*ib)+TINY for a in x]
    normbits=[bf16_of_real(a) for a in norm]
    gainbits=[bf16_of_real(expanded(a)*b) for a,b in zip(normbits,w)]
    gate=[silu(a) for a in z];output=[expanded(a)*b for a,b in zip(gainbits,gate)]
    outbits=[bf16_of_real(a) for a in output]
    late=[bf16_of_real(a*b*c) for a,b,c in zip(norm,w,gate)]
    offset=[bf16_of_real(quantize(expanded(a)*(1+b))*c) for a,b,c in zip(normbits,w,gate)]
    sigmoid=[bf16_of_real(expanded(a)/(1+math.exp(-b))) for a,b in zip(gainbits,z)]
    return {'squares':squares,'squares_bound':sb,'q':q,'q_bound':qb,'norm':norm,'norm_bounds':nb,
      'norm_bits':normbits,'gain_bits':gainbits,'gate':gate,'output':output,'output_bits':outbits,
      'adversaries':{'late_cast':sum(a!=b for a,b in zip(late,outbits)),
                     'offset_gain':sum(a!=b for a,b in zip(offset,outbits)),
                     'sigmoid_only':sum(a!=b for a,b in zip(sigmoid,outbits))}}


def validate(case,observed):
    x,w,z=(case[n] for n in ('x','w','z'));ref=reference(x,w,z)
    squares,q,root,inverse=observed['stats'];norm=observed['norm'];normbits=observed['norm_bits']
    gainpre=observed['gain_pre'];gainbits=observed['gain_bits'];gate=observed['gate'];pre=observed['precast'];out=observed['output_bits'];ex=observed['exp']
    def relative(actual,expected,budget):return close(actual,expected,[budget*abs(v)+TINY for v in expected])
    gates={'sum':close([squares],[ref['squares']],[ref['squares_bound']]),'denominator':close([q],[ref['q']],[ref['q_bound']]),
      'sqrt':relative([root],[math.sqrt(q)],POLICY['sqrt_relative']),
      'inverse':relative([inverse],[1/root],POLICY['inverse_relative']),
      'normalized_math':close(norm,ref['norm'],ref['norm_bounds']),
      'normalized_actual_inverse':relative(norm,[a*inverse for a in x],U),
      'gain_product':relative(gainpre,[expanded(a)*b for a,b in zip(normbits,w)],U),
      'exp':relative(ex,[math.exp(-abs(a)) for a in z],POLICY['exp_relative']),
      'silu':relative(gate,ref['gate'],POLICY['silu_relative']),
      'final_product':relative(pre,[expanded(a)*b for a,b in zip(gainbits,gate)],U)}
    conversions={n:actual==[bf16_rne(bits(v)) for v in values] for n,actual,values in (
        ('normalized',normbits,norm),('gain_product',gainbits,gainpre),('final',out,pre))}
    annihilation=all(out[i]&0x7fff==0 for i in range(N) if x[i]==0 or w[i]==0 or z[i]==0)
    return {'passed':all(v['passed'] for v in gates.values()) and all(conversions.values()) and annihilation,
       'stages':gates,'conversions':conversions,'annihilation':annihilation,
       'oracle_output_bf16_mismatches_non_gating':sum(a!=b for a,b in zip(out,ref['output_bits'])),
       'adversaries':ref['adversaries']}
