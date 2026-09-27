# Candidate exact FP8 lowering (not yet a device kernel)

The baseline decodes each weight through a scalar table before a vector FP32 FMA.
The official installed SDK `dsd_ops.csl` exposes `@fmachs` for native FP16 inputs
and FP32 accumulation. A possible exact representation avoids the per-value table:

```
half_bits = ((fp8_code & 127) << 7) | ((fp8_code & 128) << 8)
E4M3FN(fp8_code) = binary16(half_bits) * 256
```

The original 4-bit exponent occupies the low four bits of binary16's 5-bit
exponent; its bias difference is exactly eight. For exponent zero the subnormal
step changes by the same factor 256. Signed zeros are preserved. NaN encodings
127/255 remain rejected. All 254 finite encodings and 64,516 ordered products pass an
independent standard-library IEEE check in `fp8-rescaling-identity-001.json`.

With native FP16 multiplication / FP32 accumulation in the original term order,
rescaling the block dot by 65,536 before the original activation and weight scales
can preserve the original FP32 result: multiplication by a power of two commutes
with rounding in the safe exponent range of these bounded 128-term products.
This needs explicit accumulation and boundary tests, not just the encoding proof.

The critical outstanding condition is **device FP16 subnormal input handling**.
Small FP8 magnitudes map to FP16 subnormals. A flush-to-zero input policy would
invalidate the transformation. The next bounded kernel probe must test every
finite encoding (including signed zero), mixed products and actual original-weight
tiles before measuring its cycles. Packed bit extraction also needs a compiled,
measured vector implementation. No speedup or device correctness is asserted here.

`fp8-native-sim-001` has now passed all 64,516 ordered finite products through
`@fmachs`, including FP16 subnormal inputs, followed by two 128-repeat accumulations
and weight retention. This is WSE-3 simulator evidence. Each 254-element repeated
FMA vector took 135.09375 simulator cycles including its loop. The physical probe
is staged separately; these observations are not physical timing or a packed
FP8 GEMV result.

The subsequent `fp8-native-hw-001` physical probe also passed all 64,516 products,
the two repeated-accumulation cases and retention with normal resource release.
This establishes the required subnormal behavior for the tested multiply path.
Packed decode, variable-sign dot sums with original model tiles, overlap and
full-model output qualification remain outstanding.
