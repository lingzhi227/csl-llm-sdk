"""Exact rational checks for all five observed Q/K RMS statistics.

Uses the already declared contracts: gamma(512) for256 square/accumulates,
u for scalar FP32 arithmetic, and2^-20 for sqrt/reciprocal approximations.
This checks those contracts; it does not establish a stronger SDK math bound.
No overwritten FP32 normalized/product value is treated as an observation.
"""
try:
    F, U, TINY, f32_value, bf16_value, normal_or_zero, require
except NameError:
    from .qk_alias_enclosure import (F, U, TINY, f32_value, bf16_value,
                                   normal_or_zero, require)

EPSILON_BITS = 0x358637BD
APPROX = F(1, 1 << 20)


def check_retained_stats(input_bits, stat_bits):
    require(len(input_bits) == 512 and len(stat_bits) == 10,
            "two256-channel heads and ten observed FP32 statistics")
    require(all(normal_or_zero(x, 16) for x in input_bits),
            "normal-or-zero original BF16 inputs")
    inputs = [bf16_value(x) for x in input_bits]
    observed = [f32_value(x) for x in stat_bits]
    epsilon = f32_value(EPSILON_BITS)
    results = []
    gamma512 = 512 * U / (1 - 512 * U)

    def interval_check(actual, expected, error):
        return dict(passed=abs(actual - expected) <= error,
                    absolute_error=float(abs(actual - expected)),
                    allowed_absolute_error=float(error))

    for h in range(2):
        s = observed[h * 5:(h + 1) * 5]
        domain = (s[0] >= 0 and s[1] >= 0 and s[2] > 0
                  and s[3] > 0 and s[4] > 0
                  and normal_or_zero(stat_bits[h * 5 + 3], 32)
                  and normal_or_zero(stat_bits[h * 5 + 4], 32))
        require(domain, "nonnegative moments and positive normal sqrt/inverse")
        square = sum((v * v for v in inputs[h * 256:(h + 1) * 256]), F(0))
        mean = s[0] / 256
        denominator = s[1] + epsilon
        inverse = 1 / s[3]
        # Exact equivalent of |root-sqrt(z)| <= APPROX*sqrt(z)+TINY.
        # All terms are nonnegative, so squaring preserves the inequalities.
        lower_ok = (s[3] + TINY) ** 2 >= (1 - APPROX) ** 2 * s[2]
        upper_ok = max(s[3] - TINY, F(0)) ** 2 <= (1 + APPROX) ** 2 * s[2]
        checks = {
            "square": interval_check(s[0], square, gamma512 * square + 512 * TINY),
            "mean": interval_check(s[1], mean, U * abs(mean) + TINY),
            "denominator": interval_check(s[2], denominator,
                                           U * abs(denominator) + TINY),
            "root": dict(passed=lower_ok and upper_ok,
                         lower_squared_inequality=lower_ok,
                         upper_squared_inequality=upper_ok),
            "inverse": interval_check(s[4], inverse, APPROX * inverse + TINY),
        }
        results.append(dict(head=h, passed=all(v["passed"] for v in checks.values()),
                            checks=checks))
    return dict(passed=all(h["passed"] for h in results), heads=results,
                epsilon_fp32_bits=EPSILON_BITS,
                observed_statistics=10, overwritten_products_used=False,
                comparison_arithmetic="exact rational; exact squared sqrt inequalities")
