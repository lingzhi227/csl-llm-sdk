"""WP05 stage contracts and independent finite-domain BF16 rounding oracle."""
import math
import struct
N = 5120
U = 2**-24
TINY = 2**-126
def f32(x): return struct.unpack('<f', struct.pack('<f', x))[0]
def bits(x): return struct.unpack('<I', struct.pack('<f', x))[0]
def from_bits(x): return struct.unpack('<f', struct.pack('<I', x))[0]
EPSILON = f32(1e-6)
POLICY = {'name':'ordinary-rms5120-offset-gain-v1', 'columns':N, 'epsilon_f32':EPSILON,
          'unit_roundoff':U, 'sum_bound':'gamma(2*N)*sum(x*x)+2*N*min_normal',
          'denominator_bound':'sum_bound/N + gamma(2)*(abs(sum/N)+epsilon+sum_bound/N)+2*min_normal',
          'sqrt_relative_budget':2**-20, 'reciprocal_relative_budget':2**-20,
          'stage_gates':'independent sqrt(actual denominator), reciprocal(actual sqrt), gain application using actual inverse',
          'gain':'BF16 exact expansion, FP32(1+w), then FP32(FP32(x*inverse)*(1+w))',
          'output':'exact BF16 RNE of actual pre-cast FP32 bits; mathematical-oracle BF16 parity reported separately',
          'domain':'synthetic finite normal BF16 inputs/gains or zero; bounded products, no overflow/subnormal result claim',
          'in_place':'FP32 staging overwritten with pre-cast; BF16 gain slot overwritten with output after gain consumption; both host originals remain immutable; next call reloads full gain',
          'conversion_probe_scope':'16 finite normal signed midpoint/adjacent/zero patterns; not unrestricted NaN/subnormal support'}
PROBE_BITS = [0x3f808000,0x3f818000,0xbf808000,0xbf818000,0x3f807fff,0x3f808001,
              0xbf807fff,0xbf808001,0,0x80000000,0x3f800000,0xbf800000,0x3fff8000,
              0xbfff8000,0x40008000,0xc0008000]


def bf16_rne(raw):
    """Independent nearest-neighbor distance rule, not the device integer-bias trick."""
    if type(raw) is not int or not 0 <= raw < 2**32 or raw & 0x7f800000 == 0x7f800000:
        raise ValueError('Expected finite FP32 bits')
    hi, lo = raw >> 16, raw & 65535
    if lo < 32768: return hi
    if lo > 32768: return (hi+1) & 65535
    return hi if hi % 2 == 0 else (hi+1) & 65535


def gamma(n): return n*U/(1-n*U)


def bf16_of_real(value):
    """Nearest finite BF16 to the independent FP64 value, avoiding double rounding."""
    if not math.isfinite(value): raise ValueError('Nonfinite oracle')
    if value == 0: return bits(value) >> 16
    center=bits(value)>>16
    candidates=[b for b in (center-1,center,center+1) if 0<=b<=65535 and b&0x7f80!=0x7f80]
    return min(candidates,key=lambda b:(abs(from_bits(b<<16)-value),b&1))


def fixtures():
    data = [
        ('zero', [0.0]*N, [0.0]*N),
        ('unit_gain', [(j%29-14)/8 for j in range(N)], [0.0]*N),
        ('signed_offset_gain', [((j*3)%31-15)/16 for j in range(N)], [(j%9-4)/4 for j in range(N)]),
        ('rounding_boundary', [((j*17)%127-63)/32 for j in range(N)], [(j%17-8)/32 for j in range(N)]),
    ]
    for name, x, w in data:
        if any(bits(v)&65535 for v in x+w): raise ValueError('Fixture not exactly BF16 representable')
        yield name, x, w


def reference(x,w):
    if len(x)!=N or len(w)!=N or any(not math.isfinite(v) for v in x+w): raise ValueError('Invalid fixture')
    squares=math.fsum(v*v for v in x)
    sb=gamma(2*N)*squares+2*N*TINY
    q=squares/N+EPSILON
    qb=sb/N+gamma(2)*(abs(squares/N)+EPSILON+sb/N)+2*TINY
    low=max(EPSILON*(1-U),q-qb)
    root=math.sqrt(q);lo_root=math.sqrt(low)
    approximation=(1+POLICY['reciprocal_relative_budget'])/(1-POLICY['sqrt_relative_budget'])-1
    inverse_bound=qb/(lo_root*root*(lo_root+root))+approximation/lo_root
    expected=[];bounds=[]
    for a,g in zip(x,w):
        gain=1+g;gain_error=U*abs(gain)+TINY
        y=a/root*gain
        before=abs(a)*inverse_bound*abs(gain)+abs(a)/lo_root*(1+approximation)*gain_error
        bound=before+gamma(2)*(abs(y)+before)+4*TINY
        expected.append(y);bounds.append(bound)
    return {'squares':squares,'squares_bound':sb,'q':q,'q_bound':qb,'expected':expected,'bounds':bounds}


def close(actual,expected,bounds):
    if not actual or len(actual)!=len(expected) or len(bounds)!=len(actual): raise ValueError('Shape mismatch')
    if any(not math.isfinite(x) for seq in (actual,expected,bounds) for x in seq) or any(b<=0 for b in bounds):raise ValueError('Nonfinite/invalid gate')
    errors=[abs(a-e) for a,e in zip(actual,expected)]
    return {'passed':all(e<=b for e,b in zip(errors,bounds)), 'max_absolute_error':max(errors),
            'max_error_over_bound':max(e/b for e,b in zip(errors,bounds))}


def gain_gate(x,w,inverse,actual):
    expected=[a*inverse*(1+g) for a,g in zip(x,w)]
    bounds=[gamma(3)*abs(v)+4*TINY for v in expected]
    return close(actual,expected,bounds)
