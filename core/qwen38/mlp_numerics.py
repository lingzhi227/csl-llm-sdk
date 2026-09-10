"""WP16 source-only SiLU/product/operand enclosures; no model or SDK execution."""
import math
import struct

from .trained_qk_rope_numerics import Interval, down, fp32, up

U32 = 2.0 ** -24
U64 = 2.0 ** -53
FP32_TINY = 2.0 ** -126
BF16_TINY = 2.0 ** -133
LIBM_RELATIVE = 2.0 ** -44
SIGMOID_RELATIVE = 2.0 ** -18
BF16_OVERFLOW_THRESHOLD = float.fromhex('0x1.ffp127')


def round_scalar(value):
    """Integer RNE from binary64 bits; no intermediate FP32 or denormal-mode dependence."""
    if not math.isfinite(value) or abs(value) >= BF16_OVERFLOW_THRESHOLD:
        raise ValueError('Source interval reaches BF16 rounding overflow')
    raw = struct.unpack('<Q', struct.pack('<d', value))[0]
    exponent, fraction = (raw >> 52) & 2047, raw & ((1 << 52) - 1)
    significand = fraction if exponent == 0 else (1 << 52) | fraction
    power = -1074 if exponent == 0 else exponent - 1023 - 52
    if significand == 0:
        return -0. if raw >> 63 else 0.
    scale = max(significand.bit_length() - 1 + power - 7, -133)
    shift = power - scale
    if shift >= 0:
        integer = significand << shift
    else:
        divisor = 1 << -shift
        integer, remainder = divmod(significand, divisor)
        if 2 * remainder > divisor or (2 * remainder == divisor and integer & 1):
            integer += 1
    rounded = math.ldexp(float(integer), scale)
    return -rounded if raw >> 63 else rounded


def bf16_round(value):
    # The older finite-domain oracle excludes infinities; reject rather than clamp
    # an interval reaching the BF16 RNE overflow midpoint (the tie rounds to inf).
    if value.magnitude >= BF16_OVERFLOW_THRESHOLD:
        raise ValueError('Source interval reaches BF16 rounding overflow')
    return Interval(round_scalar(value.lo), round_scalar(value.hi))


def bf16_code(value):
    rounded = round_scalar(value)
    if struct.pack('<d', rounded) != struct.pack('<d', float(value)):
        raise ValueError('Exact BF16 value required')
    sign = 0x8000 if math.copysign(1., value) < 0 else 0
    magnitude = abs(value)
    if magnitude == 0:
        return sign
    if magnitude < FP32_TINY:
        return sign | int(math.ldexp(magnitude, 133))
    exponent = math.frexp(magnitude)[1] - 1
    mantissa = int(math.ldexp(magnitude, 7 - exponent)) - 128
    return sign | ((exponent + 127) << 7) | mantissa


def require_bf16(value):
    bf16_code(value.lo)
    bf16_code(value.hi)


def effective_operand(value):
    """Enclose both preserved and flushed nonzero-subnormal BF16 operands."""
    require_bf16(value)
    largest_subnormal = FP32_TINY - BF16_TINY
    positive = value.lo <= largest_subnormal and value.hi >= BF16_TINY
    negative = value.lo <= -BF16_TINY and value.hi >= -largest_subnormal
    if positive or negative:
        return Interval(min(value.lo, 0.0), max(value.hi, 0.0)), True
    return value, False


def ordered_bf16(value):
    code = bf16_code(value)
    magnitude = code & 0x7fff
    # Numeric zero is counted once; raw signed-zero comparisons remain separate.
    return 0x8000 - magnitude if code & 0x8000 else 0x8000 + magnitude


def span_count(value):
    require_bf16(value)
    return ordered_bf16(value.hi) - ordered_bf16(value.lo) + 1


def gamma(operations, unit_roundoff):
    if type(operations) is not int or operations < 1 or operations * unit_roundoff >= 1:
        raise ValueError('Invalid reduction operation count')
    product = up(operations * unit_roundoff)
    return up(product / down(1.0 - product))


def reduction_error(n, magnitude_upper):
    """FP32 serial-product/add bound, including amplified result-FTZ loss."""
    if type(n) is not int or not 1 <= n <= 17408 or not math.isfinite(magnitude_upper) or magnitude_upper < 0:
        raise ValueError('Invalid full-contraction magnitude/count')
    operations = 2 * n
    rounding = up(gamma(operations, U32) * magnitude_upper)
    loss = up(up(operations * FP32_TINY) / down(1.0 - up(operations * U32)))
    return up(rounding + loss)


def silu(gate):
    """Global5/4 derivative enclosure plus qualified pre-cast sigmoid and product error."""
    gate, operand_ftz_possible = effective_operand(gate)
    if gate.magnitude > 24:
        raise ValueError('Source gate cannot prove exp argument in[-24,0]')
    if gate.zero:
        zero = Interval.point(0)
        return dict(effective_gate=gate, ideal=zero, fp32=zero, bf16=zero,
                    operand_ftz_possible=operand_ftz_possible)
    center = gate.lo / 2.0 + gate.hi / 2.0
    radius = max(up(center - gate.lo), up(gate.hi - center))
    exponential = math.exp(-abs(center))
    exp_interval = Interval(down(exponential), up(exponential)) * Interval(
        1.0 - LIBM_RELATIVE, 1.0 + LIBM_RELATIVE)
    inverse = (Interval.point(1) + exp_interval).reciprocal()
    sigmoid = inverse if center >= 0 else exp_interval * inverse
    evaluated = Interval.point(center) * sigmoid
    ideal = evaluated.widen(up(1.25 * radius))
    relative = up(up((1.0 + SIGMOID_RELATIVE) * (1.0 + U32)) - 1.0)
    arithmetic = up(up(gate.magnitude * relative) + FP32_TINY)
    candidate = ideal.widen(arithmetic)
    return dict(effective_gate=gate, ideal=ideal, fp32=candidate, bf16=bf16_round(candidate),
                operand_ftz_possible=operand_ftz_possible)


def intermediate_product(silu_bf16, up_bf16):
    left, left_ftz = effective_operand(silu_bf16)
    right, right_ftz = effective_operand(up_bf16)
    product = fp32(left * right)
    return dict(fp32=product, bf16=bf16_round(product),
                silu_operand_ftz_possible=left_ftz, up_operand_ftz_possible=right_ftz)
