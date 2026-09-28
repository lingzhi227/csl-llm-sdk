"""Independent full5120 residual/RMS reference, without candidate CSL imports.

The explicit distributed FP32 order is compared to a separate FP64 expression
before rounding. The conservative gamma bound covers an ordinary128-term sum,
39 cross-shard additions, division/sqrt/reciprocal and normalization products;
it does not depend on assuming a benefit from the compensated local sum.
This reference supports finite normal-range fixtures, not arbitrary infinities
or underflow. It is numerical evidence only, never model/device execution.
"""
import numpy as np
from mlp_oracle import bf16_bits, bf16_float


def residual_rms(left, right, gain):
    left, right, gain = [np.asarray(x) for x in (left, right, gain)]
    if any(x.dtype != np.uint16 or x.shape != (5120,) for x in (left, right, gain)):
        raise ValueError('Original5120 BF16 operands and gain required')
    residual = bf16_bits(np.float32(bf16_float(left) + bf16_float(right)))
    x = bf16_float(residual); g = bf16_float(gain)
    if not np.isfinite(x).all() or not np.isfinite(g).all() or np.max(np.abs(x)) > 2**30:
        raise ValueError('Reference finite bounded domain')
    local = np.zeros(40, np.float32); correction = np.zeros(40, np.float32)
    values = x.reshape(40, 128)
    for i in range(128):
        term = np.float32(np.float32(values[:, i] * values[:, i]) - correction)
        updated = np.float32(local + term)
        correction = np.float32(np.float32(updated - local) - term); local = updated
    total = local[-1]
    for i in range(38, -1, -1): total = np.float32(local[i] + total)
    inverse = np.float32(1 / np.sqrt(np.float32(np.float32(total / np.float32(5120)) + np.float32(1e-6))))
    ordered = np.float32(np.float32(x * inverse) * np.float32(np.float32(1) + g))
    mathematical = (x.astype(np.float64) / np.sqrt(np.mean(x.astype(np.float64)**2) + 1e-6)) * (1 + g.astype(np.float64))
    operations = 128 + 39 + 20; gamma = operations * 2**-24 / (1 - operations * 2**-24)
    bound = gamma * np.abs(mathematical) + np.finfo(np.float32).tiny
    if not np.all(np.abs(ordered.astype(np.float64) - mathematical) <= bound):
        raise ValueError('Ordered distributed RMS exceeds independent FP64 bound')
    return dict(residual=residual, normalized=bf16_bits(ordered), fp32=ordered,
                fp64=mathematical, bound=bound, partials=local, inverse=inverse,
                bf16_fp64=bf16_bits(mathematical))
