"""WP14 source enclosure from frozen original-hidden projection intervals.

This is independent of observed producer or consumer values. Older fixture
policies remain unchanged. Intervals deliberately forget correlations, which
can widen a result but cannot justify narrowing it using SDK observations.
"""
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import bisect
import math

from .rms_numerics import f32, EPSILON, bits
from .gated_numerics import expanded
from .interval_numerics import rational_bf16
from .rotary_trig import ABS_GATE

U = 2.0**-24
FTZ = 2.0**-126
APPROX = 2.0**-20
POLICY = {
    'name': 'original-hidden-trained-qk-rms256-rope64-v1',
    'head_dim': 256, 'rotary_dim': 64, 'positions': [1, 2, 3, 0],
    'source': 'Unchanged WP13 original-row-major FP64/serial-FMA BF16 nominal and radius; Q rows0:256, K rows512:768.',
    'domain': 'Four frozen full5120 hidden vectors and original head0 trained Q/K norm offsets; finite BF16, positive denominator; no old offset0.5 or tail3 limit.',
    'endpoint_math': 'Each FP64 endpoint operation rounds out with nextafter. Zero identities are exact. Dependence between an input and its squared reduction is discarded conservatively.',
    'squares': 'Endpoint extrema, zero minimum if crossing0; outward sum256; +/-gamma512*sum_upper +512*2^-126 covers FP32 product/reduction and possible flush-to-zero.',
    'fp32_round': 'Outward real interval +/-2^-24*maximum_absolute_endpoint +2^-126, except exact structural zero.',
    'rms': 'FP32 mean256, FP32 addition of epsilon_f32, positive sqrt enclosure with relative2^-20, reciprocal with relative2^-20, FP32 normalize, FP32(1+w), FP32 gain, monotone BF16 RNE.',
    'epsilon_f32': EPSILON,
    'trig': 'Pinned32 FP32 frequencies; FP32 position product enclosure; sin/cos Lipschitz angle perturbation plus2^-20 device approximation and2^-44 FP64 libm allowance on[0,7]; independent BF16 casts.',
    'rotary': 'Split-half first64; FP32 product then BF16 for each term, FP32 sum then BF16; tail192 inherits normalized intervals unchanged.',
    'bf16': 'Exact rational endpoint RNE using monotonicity; finite values only. Numeric source intervals identify +/-zero; separate device checks retain sign bits.',
    'nominal': 'Ideal FP64 normalization/rotation of source projection nominal, BF16 at source boundaries; diagnostic only, never used to shrink intervals.',
    'zero': 'Zero projection interval remains exactly zero through products, gains and rotation; denominator/root still bounded.',
    'qualification': 'A conservative fixed-input source gate, not a general bitwise guarantee or SDK qualification.',
}


def down(x):
    return math.nextafter(x, -math.inf)


def up(x):
    return math.nextafter(x, math.inf)


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    def __post_init__(self):
        if not math.isfinite(self.lo) or not math.isfinite(self.hi) or self.lo > self.hi:
            raise ValueError('Invalid finite interval')

    @classmethod
    def point(cls, x):
        return cls(float(x), float(x))

    @property
    def zero(self):
        return self.lo == self.hi == 0.0

    @property
    def magnitude(self):
        return max(abs(self.lo), abs(self.hi))

    def __add__(self, other):
        if self.zero:
            return other
        if other.zero:
            return self
        return Interval(down(self.lo + other.lo), up(self.hi + other.hi))

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __mul__(self, other):
        if self.zero or other.zero:
            return Interval.point(0)
        products = [a*b for a in (self.lo, self.hi) for b in (other.lo, other.hi)]
        return Interval(down(min(products)), up(max(products)))

    def widen(self, error):
        if error < 0 or not math.isfinite(error):
            raise ValueError('Invalid absolute error')
        if error == 0:
            return self
        return Interval(down(self.lo-error), up(self.hi+error))

    def square(self):
        if self.zero:
            return self
        endpoints = (self.lo*self.lo, self.hi*self.hi)
        minimum = 0.0 if self.lo <= 0 <= self.hi else max(0.0, down(min(endpoints)))
        return Interval(minimum, up(max(endpoints)))

    def sqrt(self):
        if self.lo <= 0:
            raise ValueError('Strictly positive denominator required')
        return Interval(down(math.sqrt(self.lo)), up(math.sqrt(self.hi)))

    def reciprocal(self):
        if self.lo <= 0:
            raise ValueError('Strictly positive divisor required')
        return Interval(down(1/self.hi), up(1/self.lo))


def fp32(value):
    if value.zero:
        return value
    return value.widen(up(up(U*value.magnitude)+FTZ))


def bf16(value):
    return Interval(rational_bf16(Fraction.from_float(value.lo)),
                    rational_bf16(Fraction.from_float(value.hi)))


def q(value):
    return rational_bf16(Fraction.from_float(float(value)))


def input_interval(center, radius):
    if radius < 0:
        raise ValueError('Negative projection radius')
    return Interval.point(center).widen(radius)


def rms(inputs, weights, centers):
    if not len(inputs) == len(weights) == len(centers) == 256:
        raise ValueError('Full Q/K head required')
    if any(not math.isfinite(w) or q(w) != w for w in weights):
        raise ValueError('Original finite BF16 norm parameters required')
    square = sum((x.square() for x in inputs), Interval.point(0))
    if not square.zero:
        # The 512-operation bound admits separate squares/adds and fused variants.
        gamma512 = up((512*U)/(1-512*U))
        error = up(up(gamma512*square.hi)+512*FTZ)
        square = square.widen(error)
        square = Interval(max(0.0, square.lo), square.hi)
    mean = fp32(square*Interval.point(1/256))
    denominator = fp32(mean+Interval.point(EPSILON))
    root = denominator.sqrt()*Interval(1-APPROX, 1+APPROX)
    inverse = root.reciprocal()*Interval(1-APPROX, 1+APPROX)
    normalized = [fp32(x*inverse) for x in inputs]
    gains = [fp32(Interval.point(1)+Interval.point(w)) for w in weights]
    gained = [fp32(x*g) for x, g in zip(normalized, gains)]
    output = [bf16(x) for x in gained]
    ideal_inverse = 1/math.sqrt(math.fsum(x*x for x in centers)/256+EPSILON)
    nominal = [q(x*ideal_inverse*(1+w)) for x, w in zip(centers, weights)]
    return dict(stats=[square, mean, denominator, root, inverse],
                normalized=normalized, gains=gains, gained=gained,
                output=output, nominal=nominal)


def reference(centers, radii, weights, frequencies, position):
    if not len(centers) == len(radii) == len(weights) == 512 or len(frequencies) != 32:
        raise ValueError('Q256/K256 and32 inverse frequencies required')
    if type(position) is not int or not 0 <= position <= 7:
        raise ValueError('Text position outside qualified trig range')
    if any(not math.isfinite(f) or not 0 < f <= 1 or f32(f) != f for f in frequencies):
        raise ValueError('Pinned FP32 frequency domain')
    inputs = [input_interval(v, e) for v, e in zip(centers, radii)]
    angles, sines, cosines, sn, cn = [], [], [], [], []
    for frequency in frequencies:
        real_angle = position*frequency  # Integer0..7 * FP32 is exact in FP64.
        angle = fp32(Interval.point(real_angle)) if position else Interval.point(0)
        angles.append(angle)
        if position == 0:
            sine, cosine = Interval.point(0), Interval.point(1)
        else:
            angle_error = up(max(abs(angle.lo-real_angle), abs(angle.hi-real_angle)))
            error = up(up(angle_error+ABS_GATE)+2**-44)
            sine = Interval.point(math.sin(real_angle)).widen(error)
            cosine = Interval.point(math.cos(real_angle)).widen(error)
        sines.append(bf16(sine)); cosines.append(bf16(cosine))
        sn.append(q(math.sin(real_angle))); cn.append(q(math.cos(real_angle)))
    heads = []
    for head in range(2):
        sl = slice(head*256, (head+1)*256)
        nr = rms(inputs[sl], weights[sl], centers[sl])
        output = list(nr['output']); nominal = list(nr['nominal'])
        products0, products1, sums = [], [], []
        for c in range(64):
            other, sign = (c+32, -1) if c < 32 else (c-32, 1)
            # Read immutable normalized intervals; output is filled below only.
            rotated = -nr['output'][other] if sign == -1 else nr['output'][other]
            p0 = bf16(fp32(nr['output'][c]*cosines[c % 32]))
            p1 = bf16(fp32(rotated*sines[c % 32]))
            summed = fp32(p0+p1)
            products0.append(p0); products1.append(p1); sums.append(summed)
            output[c] = bf16(summed)
            n0 = q(nr['nominal'][c]*cn[c % 32])
            n1 = q(sign*nr['nominal'][other]*sn[c % 32])
            nominal[c] = q(f32(n0+n1))
        heads.append(dict(rms=nr, product0=products0, product1=products1,
                          sums=sums, output=output, nominal=nominal))
    return dict(inputs=inputs, angles=angles, sine=sines, cosine=cosines,
                sine_nominal=sn, cosine_nominal=cn, heads=heads)


@lru_cache(maxsize=1)
def finite_bf16():
    return sorted(set(expanded(i) for i in range(65536) if i & 0x7f80 != 0x7f80))


def report(actual, intervals, nominal=None):
    if len(actual) != len(intervals) or not actual:
        raise ValueError('Observation shape mismatch')
    failures = [i for i, (x, b) in enumerate(zip(actual, intervals))
                if not math.isfinite(x) or not b.lo <= x <= b.hi]
    result = dict(passed=not failures, count=len(actual), outside=len(failures),
                  first_outside=failures[:8], max_width=max(b.hi-b.lo for b in intervals))
    if nominal is not None:
        result.update(nominal_bf16_mismatches=sum(bits(a) != bits(n) for a, n in zip(actual, nominal)),
                      max_nominal_difference=max(abs(a-n) for a, n in zip(actual, nominal)))
    return result


def spans(intervals):
    finite = finite_bf16()
    counts = [bisect.bisect_right(finite, b.hi)-bisect.bisect_left(finite, b.lo) for b in intervals]
    if min(counts) < 1:
        raise ValueError('Expected nonempty BF16 output intervals')
    return dict(max_count=max(counts), single_value_elements=sum(n == 1 for n in counts),
                min_count=min(counts), max_numeric_span=max(b.hi-b.lo for b in intervals))
