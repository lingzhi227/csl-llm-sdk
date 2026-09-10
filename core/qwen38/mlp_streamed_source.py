"""WP16 bounded-row FP64 source builder. Execution still needs CPU admission."""
import math
import numpy as np

from .mlp_numerics import (BF16_TINY, FP32_TINY, Interval, U32, U64, bf16_round, gamma,
                         intermediate_product, silu)
from .trained_qk_rope_numerics import down, up

CASES = ('dense_dyadic', 'changed_dyadic', 'last_column_onehot', 'zero_after_nonzero')
MAX_F32 = float.fromhex('0x1.fffffep127')


def exact_bf16(values, *, weights=False):
    if values.dtype != np.float64 or not np.isfinite(values).all():
        raise ValueError('Finite FP64 expansions of source BF16 required')
    raw = values.view(np.uint64)
    magnitude = raw & 0x7fffffffffffffff
    exponent = (raw >> 52) & 2047
    nonzero = magnitude != 0
    normal = (exponent >= 897) & (exponent <= 1150)
    subnormal = (exponent >= 890) & (exponent <= 896)
    if np.any(nonzero & ~(normal | subnormal)) or np.any(normal & ((raw & ((1 << 45) - 1)) != 0)):
        raise ValueError('Source endpoints must be exact BF16, without nextafter padding')
    if weights and np.any(subnormal):
        raise ValueError('Original nonzero subnormal weight requires separate qualification')
    if np.any(subnormal):
        scaled = np.ldexp(np.abs(values[subnormal]), 133)
        if not np.equal(scaled, np.floor(scaled)).all():
            raise ValueError('Non-BF16 value in the subnormal range')


def quantize(bounds):
    result = np.empty_like(bounds)
    for i, (low, high) in enumerate(bounds):
        value = bf16_round(Interval(float(low), float(high)))
        result[i] = value.lo, value.hi
    return result


def effective(bounds):
    exact_bf16(bounds)
    low, high = bounds[:, 0].copy(), bounds[:, 1].copy()
    edge = FP32_TINY - BF16_TINY
    possible = ((low <= edge) & (high >= BF16_TINY)) | ((low <= -BF16_TINY) & (high >= -edge))
    low[possible] = np.minimum(low[possible], 0.)
    high[possible] = np.maximum(high[possible], 0.)
    return low, high, int(np.count_nonzero(possible))


def linear_bounds(weights, bounds):
    """At most64 rows; exact endpoint products, outward FP64/FP32 sum bounds."""
    if weights.ndim != 2 or not 1 <= weights.shape[0] <= 64:
        raise ValueError('Use a1..64-row weight chunk')
    n = weights.shape[1]
    if not 1 <= n <= 17408 or bounds.shape != (n, 2) or np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError('Contraction dimension/interval mismatch')
    exact_bf16(weights, weights=True)
    low, high, ftz_count = effective(bounds)
    # BF16 endpoint products have at most16 significant bits and fit normal FP64.
    left = weights * low
    right = weights * high
    lower_sum = np.minimum(left, right).sum(axis=1, dtype=np.float64)
    upper_sum = np.maximum(left, right).sum(axis=1, dtype=np.float64)
    np.abs(left, out=left)
    np.abs(right, out=right)
    np.maximum(left, right, out=left)
    magnitude_sum = left.sum(axis=1, dtype=np.float64)
    del left, right
    exact_zero = magnitude_sum == 0.
    outward_up = lambda x: np.nextafter(x, np.inf)
    outward_down = lambda x: np.nextafter(x, -np.inf)
    g64 = gamma(n, U64)
    magnitude_upper = outward_up(magnitude_sum / down(1. - g64))
    fp64_error = outward_up(g64 * magnitude_upper)
    g32 = gamma(2 * n, U32)
    ftz_error = up(up(2 * n * FP32_TINY) / down(1. - up(2 * n * U32)))
    fp32_error = outward_up(outward_up(g32 * magnitude_upper) + ftz_error)
    if np.any(outward_up(magnitude_upper + fp32_error) > MAX_F32):
        raise ValueError('Source contraction cannot prove absence of FP32 overflow')
    result = np.column_stack((outward_down(outward_down(lower_sum - fp64_error) - fp32_error),
                              outward_up(outward_up(upper_sum + fp64_error) + fp32_error)))
    result[exact_zero] = 0.
    magnitude_upper[exact_zero] = 0.
    return dict(fp32=result, bf16=quantize(result), magnitude_upper=magnitude_upper,
                computed_lower_sum_fp64=lower_sum, computed_upper_sum_fp64=upper_sum,
                structural_zero_rows=int(np.count_nonzero(exact_zero)),
                input_ftz_possible_count=ftz_count)


def build_source(row_reader, inputs, phase_callback=lambda phase: None):
    """Full dimensions using an original-weight FP64 row reader, never CPU observations.

    row_reader(role,start,end) must return exact original BF16 values expanded to
    FP64. Its underlying immutable segment identity/hash verification is a separate
    required loader gate. This function does not acquire or load full matrices.
    """
    if tuple(inputs) != CASES:
        raise ValueError('Exact four-case source order required')
    for hidden in inputs.values():
        if hidden.shape != (5120,):
            raise ValueError('Full original module-input width required')
        exact_bf16(hidden)
    states = {name: {} for name in CASES}
    input_bounds = {name: np.column_stack((hidden, hidden)) for name, hidden in inputs.items()}
    shortcuts = {}
    for name, hidden in inputs.items():
        nonzero = np.flatnonzero(hidden)
        if len(nonzero) == 0:
            shortcuts[name] = ('zero', None)
        elif len(nonzero) == 1 and hidden[nonzero[0]] == 1.:
            shortcuts[name] = ('onehot', int(nonzero[0]))
        else:
            shortcuts[name] = ('general', None)
    for role, prefix in (('gate_proj', 'gate'), ('up_proj', 'up')):
        phase_callback('source_' + role)
        for name in CASES:
            states[name][prefix + '_fp32'] = np.empty((17408, 2), dtype=np.float64)
            states[name][prefix + '_bf16'] = np.empty((17408, 2), dtype=np.float64)
            states[name]['nominal_' + prefix] = np.empty(17408, dtype=np.float64)
        for first in range(0, 17408, 64):
            last = min(first + 64, 17408)
            rows = row_reader(role, first, last)
            if rows.shape != (last - first, 5120):
                raise ValueError('Original projection row shape changed')
            exact_bf16(rows, weights=True)
            for name in CASES:
                kind, column = shortcuts[name]
                if kind == 'zero':
                    bounds = np.zeros((last - first, 2), dtype=np.float64)
                    rounded = bounds
                    nominal = bounds[:, 0]
                elif kind == 'onehot':
                    bounds = np.column_stack((rows[:, column], rows[:, column]))
                    rounded = bounds
                    nominal = rows[:, column]
                else:
                    result = linear_bounds(rows, input_bounds[name])
                    bounds, rounded = result['fp32'], result['bf16']
                    if result['input_ftz_possible_count'] or not np.array_equal(result['computed_lower_sum_fp64'], result['computed_upper_sum_fp64']):
                        raise ValueError('Original projection nominal requires exact non-FTZ point inputs')
                    nominal = np.array([bf16_round(Interval.point(float(v))).lo
                                        for v in result['computed_lower_sum_fp64']], dtype=np.float64)
                states[name][prefix + '_fp32'][first:last] = bounds
                states[name][prefix + '_bf16'][first:last] = rounded
                states[name]['nominal_' + prefix][first:last] = nominal
            del rows
    phase_callback('source_silu_and_product')
    for name in CASES:
        data = states[name]
        for stage in ('silu_fp32', 'silu_bf16', 'product_fp32', 'product_bf16'):
            data[stage] = np.empty((17408, 2), dtype=np.float64)
        counts = dict(gate_operand_ftz=0, silu_operand_ftz=0, up_operand_ftz=0)
        data['nominal_silu'] = np.empty(17408, dtype=np.float64)
        data['nominal_product'] = np.empty(17408, dtype=np.float64)
        for i in range(17408):
            gate = Interval(*map(float, data['gate_bf16'][i]))
            upper = Interval(*map(float, data['up_bf16'][i]))
            activation = silu(gate)
            product = intermediate_product(activation['bf16'], upper)
            for prefix, result in (('silu', activation), ('product', product)):
                for dtype in ('fp32', 'bf16'):
                    value = result[dtype]
                    data[prefix + '_' + dtype][i] = value.lo, value.hi
            counts['gate_operand_ftz'] += activation['operand_ftz_possible']
            counts['silu_operand_ftz'] += product['silu_operand_ftz_possible']
            counts['up_operand_ftz'] += product['up_operand_ftz_possible']
            g, u = float(data['nominal_gate'][i]), float(data['nominal_up'][i])
            e = math.exp(-abs(g))
            s = 1. / (1. + e) if g >= 0 else e / (1. + e)
            activation_nominal = bf16_round(Interval.point(g * s)).lo
            data['nominal_silu'][i] = activation_nominal
            data['nominal_product'][i] = bf16_round(Interval.point(activation_nominal * u)).lo
        data['operand_ftz_counts'] = counts
        # Gate source domain was checked by silu before any official reference call.
        data['source_gate_magnitude_upper'] = float(np.max(np.abs(data['gate_bf16'])))
    phase_callback('source_full_down')
    down_zero = {name: bool(np.all(states[name]['product_bf16'] == 0.)) for name in CASES}
    for name in CASES:
        allocate = np.zeros if down_zero[name] else np.empty
        states[name]['down_fp32'] = allocate((5120, 2), dtype=np.float64)
        states[name]['down_bf16'] = allocate((5120, 2), dtype=np.float64)
        states[name]['nominal_down'] = allocate(5120, dtype=np.float64)
        states[name]['down_input_ftz_possible_count'] = 0
    nominal_names = [name for name in CASES if not down_zero[name]]
    nominal_columns = np.column_stack([states[name]['nominal_product'] for name in nominal_names])
    for first in range(0, 5120, 64):
        last = first + 64
        rows = row_reader('down_proj', first, last)
        if rows.shape != (64, 17408):
            raise ValueError('Full original down dimension required')
        exact_bf16(rows, weights=True)
        # Diagnostic FP64 sums only. These never set or narrow source intervals.
        nominal_sums = rows @ nominal_columns
        for column, name in enumerate(nominal_names):
            states[name]['nominal_down'][first:last] = [
                bf16_round(Interval.point(float(v))).lo for v in nominal_sums[:, column]]
        for name in CASES:
            if down_zero[name]:
                continue
            result = linear_bounds(rows, states[name]['product_bf16'])
            states[name]['down_fp32'][first:last] = result['fp32']
            states[name]['down_bf16'][first:last] = result['bf16']
            states[name]['down_input_ftz_possible_count'] = result['input_ftz_possible_count']
        del rows
    phase_callback('source_complete_before_official_observations')
    return states
