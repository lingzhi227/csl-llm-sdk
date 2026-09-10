"""Small WP16 evidence helpers; these do not load parameters or invoke a model."""
import math
import numpy as np

from .mlp_numerics import (FP32_TINY, Interval, bf16_code, intermediate_product,
                         round_scalar, silu, span_count)
from .mlp_streamed_source import exact_bf16


def decode(words):
    """Expand BF16 bits directly to binary64, including signed zero/subnormals."""
    if words.dtype != np.dtype('<u2'):
        raise ValueError('Little-endian unsigned BF16 words required')
    exponent = ((words >> 7) & 255).astype(np.int32)
    if np.any(exponent == 255):
        raise ValueError('Nonfinite BF16 observation')
    fraction = (words & 127).astype(np.int32)
    significant = np.where(exponent == 0, fraction, fraction + 128)
    power = np.where(exponent == 0, -133, exponent - 134)
    values = np.ldexp(significant.astype(np.float64), power)
    return np.copysign(values, np.where(words & 0x8000, -1., 1.))


def encode(values):
    exact_bf16(values)
    return np.array([bf16_code(float(v)) for v in values.flat], dtype='<u2').reshape(values.shape)


def rounded(values):
    return np.array([round_scalar(float(v)) for v in values.flat], dtype=np.float64).reshape(values.shape)


def outside(bounds, observed):
    if bounds.shape != (observed.size, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError('Finite ordered endpoint pairs required')
    exact_bf16(observed)
    return (observed < bounds[:, 0]) | (observed > bounds[:, 1])


def summary(bounds, observed, nominal=None):
    exact_bf16(bounds)
    rejected = outside(bounds, observed)
    spans = np.array([span_count(Interval(float(a), float(b))) for a, b in bounds], dtype=np.int64)
    widths = bounds[:, 1] - bounds[:, 0]
    norm = float(np.linalg.norm(observed))
    result = dict(count=int(observed.size), rejected=int(np.count_nonzero(rejected)),
                  maximum_width=float(np.max(widths)),
                  width_percentiles=dict(zip(('p50', 'p90', 'p99'), map(float, np.percentile(widths, [50, 90, 99])))),
                  maximum_bf16_span=int(np.max(spans)),
                  bf16_span_percentiles=dict(zip(('p50', 'p90', 'p99'), map(float, np.percentile(spans, [50, 90, 99])))),
                  exact_numeric_zero_intervals=int(np.count_nonzero(np.all(bounds == 0., axis=1))),
                  nonzero_singleton_intervals=int(np.count_nonzero((spans == 1) & (bounds[:, 0] != 0.))),
                  nonsingleton_intervals=int(np.count_nonzero(spans > 1)),
                  width_l2_to_observed_l2=None if norm == 0 else float(np.linalg.norm(widths) / norm),
                  observed_zero_count=int(np.count_nonzero(observed == 0.)),
                  observed_negative_zero_count=int(np.count_nonzero((observed == 0.) & np.signbit(observed))))
    if nominal is not None:
        if nominal.shape != observed.shape:
            raise ValueError('Nominal shape mismatch')
        bits, expected = encode(observed), encode(nominal)
        result.update(nominal_bf16_bit_mismatches=int(np.count_nonzero(bits != expected)),
                      nominal_numeric_mismatches=int(np.count_nonzero(observed != nominal)),
                      nominal_signed_zero_mismatches=int(np.count_nonzero((observed == 0.) & (nominal == 0.) & (bits != expected))))
    return result


def conditional_nonlinear(gate, activation, upper, product):
    """Observed BF16 operands only; never use these to narrow source intervals."""
    for values in (gate, activation, upper, product):
        exact_bf16(values)
        if values.shape != gate.shape or values.ndim != 1:
            raise ValueError('Equal one-dimensional intermediate arrays required')
    silu_bounds, product_bounds = np.empty((gate.size, 2)), np.empty((gate.size, 2))
    cast_expected = np.zeros(gate.size, dtype='<u2')
    cast_qualified = np.zeros(gate.size, dtype=np.bool_)
    counts = dict(gate_ftz_possible=0, activation_ftz_possible=0, up_ftz_possible=0)
    for i, (g, a, u) in enumerate(zip(gate, activation, upper)):
        activation_result = silu(Interval.point(float(g)))
        product_result = intermediate_product(Interval.point(float(a)), Interval.point(float(u)))
        silu_bounds[i] = activation_result['bf16'].lo, activation_result['bf16'].hi
        product_bounds[i] = product_result['bf16'].lo, product_result['bf16'].hi
        counts['gate_ftz_possible'] += activation_result['operand_ftz_possible']
        counts['activation_ftz_possible'] += product_result['silu_operand_ftz_possible']
        counts['up_ftz_possible'] += product_result['up_operand_ftz_possible']
        multiplication = float(a) * float(u)
        # BF16 operands have <=8 significant bits each. Their product is exactly
        # FP32 only in this proved normal/zero domain; no native accumulator is observed.
        inputs_normal = (a == 0. or abs(a) >= FP32_TINY) and (u == 0. or abs(u) >= FP32_TINY)
        result_normal = multiplication == 0. or FP32_TINY <= abs(multiplication) <= float.fromhex('0x1.fffffep127')
        if inputs_normal and result_normal:
            cast_qualified[i] = True
            cast_expected[i] = bf16_code(round_scalar(multiplication))
    bits = encode(product)
    mismatch = cast_qualified & (bits != cast_expected)
    return dict(silu_bf16=silu_bounds, product_bf16=product_bounds,
                exact_product_cast_expected_bits=cast_expected,
                exact_product_cast_qualified=cast_qualified,
                report=dict(operand_ftz_counts=counts,
                            silu=summary(silu_bounds, activation),
                            product=summary(product_bounds, product),
                            exact_product_cast_qualified=int(np.count_nonzero(cast_qualified)),
                            exact_product_cast_excluded=int(np.count_nonzero(~cast_qualified)),
                            exact_product_cast_bit_mismatches=int(np.count_nonzero(mismatch)),
                            native_fp32_precast_observations_available=False))


def wrong_intermediates(gate, activation, upper, product):
    """Frozen five changed-input mutations; values are canonical BF16 diagnostics."""
    for value in (gate, activation, upper, product):
        exact_bf16(value)
        if value.shape != (17408,):
            raise ValueError('All original intermediate channels required')
    sigmoid = np.array([1. / (1. + math.exp(-float(g))) for g in gate], dtype=np.float64)
    omitted = product.copy()
    omitted[17280:17408] = 0.
    return dict(omitted_silu=rounded(gate * upper),
                sigmoid_substitution=rounded(rounded(sigmoid) * upper),
                missing_up=activation.copy(), wrong_channel=np.roll(product, 128),
                omitted_last_tile=omitted)
