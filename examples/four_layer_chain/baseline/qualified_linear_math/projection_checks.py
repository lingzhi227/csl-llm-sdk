"""Exact selected functions from accepted Layer3 all-operator audit."""
import math
import numpy as np
from .common import require
from .rms_numerics import U,TINY,EPSILON,gamma

def decode(value):
    return (np.asarray(value, dtype=np.uint32) << 16).view(np.float32)

def encode(value):
    raw = np.asarray(value, dtype=np.float32).view(np.uint32)
    require(np.isfinite(value).all(), 'Finite BF16 cast input')
    return ((raw + np.uint32(32767) + ((raw >> 16) & 1)) >> 16).astype(np.uint16)

def equal(a, b, message):
    require(a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(), message)

def normal(value, message):
    raw = np.asarray(value, dtype=np.float32).view(np.uint32)
    require(np.isfinite(value).all() and not np.any(
        (((raw >> 23) & 255) == 0) & ((raw & 0x7fffffff) != 0)), message)

def norm_gate(raw, gains, stats, output):
    x = decode(raw); w = decode(gains); normal(x, 'Norm finite normal inputs')
    square, q, root, inverse = map(float, stats)
    exact = math.fsum(float(v)*float(v) for v in x)
    require(abs(square-exact) <= math.nextafter(gamma(10240)*exact+10240*TINY, math.inf),
            'Norm original-input squared sum enclosure')
    wanted_q = np.float32(np.float32(square/5120)+np.float32(EPSILON))
    require(q == float(wanted_q) and q > 0 and root > 0 and inverse > 0,
            'Norm measured sum division and epsilon')
    require(abs(root-math.sqrt(q)) <= 2**-20*math.sqrt(q)+TINY, 'Norm sqrt approximation')
    require(abs(inverse-1/root) <= 2**-20/root+TINY, 'Norm inverse approximation')
    gained = np.multiply(np.multiply(x, np.float32(inverse), dtype=np.float32),
        np.add(np.float32(1), w, dtype=np.float32), dtype=np.float32)
    normal(gained, 'Norm gain arithmetic normal or zero')
    equal(output, encode(gained), 'Norm output exact FP32 replay/BF16 cast from measured inverse')
    return dict(squared_sum_absolute_error=abs(square-exact), unobserved_precast_replayed=True)

def nonlinear_gate(peers):
    g = decode(peers['mlp_gate']).reshape(-1); u = decode(peers['mlp_up']).reshape(-1)
    require(np.all(np.abs(g) <= 24) and np.all(np.abs(u) <= 24), 'Qualified original layer3 MLP domain')
    e = peers['mlp_exponential'].reshape(-1).astype(np.float64)
    sig = peers['mlp_sigmoid'].reshape(-1).astype(np.float64)
    ideal = np.exp(-np.abs(g.astype(np.float64)))
    require(np.all(np.abs(e-ideal) <= np.nextafter(ideal*(2**-20+2**-44)+TINY, np.inf)), 'MLP exp gate')
    nominal_sig = np.where(g < 0, ideal, 1)/(1+ideal)
    require(np.all(np.abs(sig-nominal_sig) <= np.nextafter(nominal_sig*(2**-18+2**-44)+TINY, np.inf)), 'MLP sigmoid gate')
    activation = np.multiply(g, peers['mlp_sigmoid'].reshape(-1), dtype=np.float32)
    equal(peers['mlp_activation'].reshape(-1), activation, 'MLP actual gate times actual sigmoid')
    equal(peers['mlp_silu'].reshape(-1), encode(activation), 'MLP complete SiLU BF16 cast')
    product = np.multiply(decode(peers['mlp_silu']).reshape(-1), u, dtype=np.float32)
    equal(peers['mlp_product_fp32'].reshape(-1), product, 'MLP BF16 SiLU times up FP32 product')
    equal(peers['mlp_product'].reshape(-1), encode(product), 'MLP product BF16 cast')
    return dict(rows=17408, exp_max_absolute_error=float(np.max(np.abs(e-ideal))))
