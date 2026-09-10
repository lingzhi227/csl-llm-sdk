"""Bounded FP32 sin/cos profile for runtime angles in[0,7]."""
import math
from fractions import Fraction
from .rms_numerics import f32,U,TINY,bits,from_bits
SIN=[Fraction((-1)**i,math.factorial(2*i+1)) for i in range(7)]
COS=[Fraction((-1)**i,math.factorial(2*i)) for i in range(7)]
HI=f32(1.5703125)
LO=f32(math.pi/2-HI)
RADIUS=0.786
ABS_GATE=2**-20


def evaluate(angle):
    x=f32(angle)
    if not 0<=x<=7:raise ValueError('Qualified angles are[0,7]')
    n=0;select=x
    while select>f32(math.pi/4):select=f32(select-f32(math.pi/2));n+=1
    r=f32(f32(x-f32(n*HI))-f32(n*LO));z=f32(r*r)
    def polynomial(cs):
        p=f32(float(cs[-1]))
        for c in reversed(cs[:-1]):p=f32(f32(float(c))+f32(z*p))
        return p
    s=f32(r*polynomial(SIN));c=polynomial(COS)
    return ((s,c),(c,-s),(-s,-c),(-c,s))[n%4],dict(quadrant=n,reduced=r,square=z)


def polynomial_bound(cs,sine):
    # Bound each unfused FP32 operation; a correctly rounded fused multiply-add
    # introduces no larger error than this separate-operation envelope.
    z=RADIUS**2;ze=U*z+TINY
    magnitude=abs(float(cs[-1]));error=abs(f32(float(cs[-1]))-float(cs[-1]))
    for c in reversed(cs[:-1]):
        coefficient=float(c);ce=abs(f32(coefficient)-coefficient)
        product_mag=z*magnitude
        product_error=z*error+ze*(magnitude+error)+U*(z+ze)*(magnitude+error)+TINY
        error=product_error+ce+U*(abs(coefficient)+ce+product_mag+product_error)+TINY
        magnitude=abs(coefficient)+product_mag
    if sine:error=RADIUS*error+U*RADIUS*(magnitude+error)+TINY
    remainder=RADIUS**(15 if sine else 14)/math.factorial(15 if sine else 14)
    return error+remainder


def analytic_budget():
    # n<=4. n*HI is exactly representable; x-n*HI is exact by Sterbenz
    # for n>0 (n=0 is identity). LO coefficient/multiply and final subtraction
    # account for reduction; derivative magnitudes of sin/cos are at most1.
    reduction=4*abs(LO-(math.pi/2-HI))+U*4*abs(LO)+U*RADIUS+8*2**-52
    sin=polynomial_bound(SIN,True)+reduction
    cos=polynomial_bound(COS,False)+reduction
    return dict(domain=[0,7],reduced_radius=RADIUS,reduction_absolute_bound=reduction,
                sin_absolute_bound=sin,cos_absolute_bound=cos,declared_absolute_gate=ABS_GATE,
                passed=max(sin,cos)<ABS_GATE,source_angle_rounding='handled separately from the exact observed FP32 angle',
                qualification='forward-error envelope plus dense scalar checks; not long-context range reduction')


def grid_check(points=28001):
    inputs={f32(7*i/(points-1)) for i in range(points)}
    for angle in (0.,7.,math.pi/4,math.pi/2,3*math.pi/4,math.pi,5*math.pi/4,3*math.pi/2,7*math.pi/4,2*math.pi):
        raw=bits(angle)
        for b in (raw-1,raw,raw+1):
            if b>=0 and 0<=from_bits(b)<=7:inputs.add(from_bits(b))
    worst=[0.,0.];max_reduced=0.
    for x in sorted(inputs):
        actual,detail=evaluate(x);errors=[abs(actual[0]-math.sin(x)),abs(actual[1]-math.cos(x))]
        worst=[max(a,b) for a,b in zip(worst,errors)];max_reduced=max(max_reduced,abs(detail['reduced']))
    envelope=analytic_budget()
    return dict(points=len(inputs),max_sin_absolute_error=worst[0],max_cos_absolute_error=worst[1],max_reduced=max_reduced,
                passed=max(worst)<ABS_GATE and max_reduced<RADIUS and envelope['passed'],analytic=envelope)
