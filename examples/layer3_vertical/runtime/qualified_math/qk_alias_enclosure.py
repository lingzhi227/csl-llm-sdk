"""Independent conditional BF16 gate for overwritten QK RMS intermediates.

Only the original BF16 input/gain, observed FP32 inverse, and observed BF16
normalization result are inputs. No reconstructed value is called observed.
The existing scalar FP32 contract is |fl(z)-z| <= 2^-24 |z| + 2^-126;
the additive term includes result flushing. Operand flushing is excluded by
requiring normal-or-zero inputs. This is an enclosure, not exact FP32 replay,
and does not validate the inverse, the subsequent rotary work, or the model.
"""
from fractions import Fraction as F

U = F(1, 1 << 24)
TINY = F(1, 1 << 126)
F32_MAX = F((1 << 24) - 1) * (1 << 104)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def power2(exponent):
    return F(1 << exponent) if exponent >= 0 else F(1, 1 << -exponent)


def f32_value(raw):
    require(type(raw) is int and 0 <= raw < 1 << 32, "FP32 word")
    exponent = (raw >> 23) & 255
    mantissa = raw & 0x7fffff
    require(exponent != 255, "finite FP32 value")
    value = (F(mantissa) * power2(-149) if exponent == 0 else
             F((1 << 23) | mantissa) * power2(exponent - 150))
    return -value if raw >> 31 else value


def bf16_value(raw):
    require(type(raw) is int and 0 <= raw < 1 << 16, "BF16 word")
    return f32_value(raw << 16)


def normal_or_zero(raw, bits):
    require(type(raw) is int and 0 <= raw < 1 << bits, "operand word")
    word = raw if bits == 32 else raw << 16
    exponent = (word >> 23) & 255
    return exponent not in (0, 255) or word & 0x7fffffff == 0


def rounded_interval(interval):
    lo, hi = interval
    require(lo <= hi, "ordered exact interval")
    if lo == hi == 0:
        return F(0), F(0)
    error = U * max(abs(lo), abs(hi)) + TINY
    lower, upper = lo - error, hi + error
    # RNE and result flushing do not reverse the sign of a scalar operation.
    if lo >= 0:
        lower = max(lower, F(0))
    if hi <= 0:
        upper = min(upper, F(0))
    require(-F32_MAX <= lower <= upper <= F32_MAX,
            "every intermediate enclosure must remain finite FP32")
    return lower, upper


def multiply(left, right):
    products = [a * b for a in left for b in right]
    return min(products), max(products)


def normalized_enclosure(input_bits, gain_bits, inverse_bits):
    require(normal_or_zero(input_bits, 16), "normal-or-zero original input")
    require(normal_or_zero(gain_bits, 16), "normal-or-zero original gain")
    require(normal_or_zero(inverse_bits, 32), "normal observed inverse")
    x, gain, inverse = (bf16_value(input_bits), bf16_value(gain_bits),
                        f32_value(inverse_bits))
    require(inverse > 0, "positive observed inverse")
    normalized = rounded_interval((x * inverse, x * inverse))
    coefficient = rounded_interval((1 + gain, 1 + gain))
    return rounded_interval(multiply(normalized, coefficient))


def bf16_cell(raw):
    """Closed numeric limits and tie inclusion for finite BF16 RNE.

    Zero signs share one numeric cell deliberately; no bitwise zero-sign claim
    is made. At the finite extrema, 2^128 is the virtual next value used for
    the overflow rounding boundary.
    """
    center = bf16_value(raw)
    magnitude = raw & 0x7fff
    if magnitude == 0:
        half_subnormal = power2(-134)
        return -half_subnormal, half_subnormal, True
    smaller = bf16_value(magnitude - 1)
    larger = power2(128) if magnitude == 0x7f7f else bf16_value(magnitude + 1)
    previous, following = (-larger, -smaller) if raw & 0x8000 else (smaller, larger)
    return (previous + center) / 2, (center + following) / 2, raw & 1 == 0


def admits_bf16(interval, raw):
    lo, hi = interval
    require(lo <= hi, "ordered candidate interval")
    left, right, inclusive = bf16_cell(raw)
    lower, upper = max(lo, left), min(hi, right)
    intersects = lower < upper or (lower == upper and
                  (inclusive or left < lower < right))
    contained = ((lo > left or inclusive and lo == left) and
                 (hi < right or inclusive and hi == right))
    return dict(passed=intersects, single_numeric_rounding_bin=contained)


def check_normalized(input_bits, gain_bits, inverse_bits, actual_bits):
    interval = normalized_enclosure(input_bits, gain_bits, inverse_bits)
    result = admits_bf16(interval, actual_bits)
    result.update(actual_bits=actual_bits, lower_exact=str(interval[0]),
                  upper_exact=str(interval[1]),
                  FP32_intermediates_observed=False,
                  zero_sign_bitwise_checked=False)
    return result
