# Exact FP8 lowering and physical qualification

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

An essential qualification condition is **device FP16 subnormal input handling**.
Small FP8 magnitudes map to FP16 subnormals. A flush-to-zero input policy would
invalidate the transformation. The bounded kernel probe must test every
finite encoding (including signed zero), mixed products and actual original-weight
tiles before measuring its cycles. Packed bit extraction also needs a compiled,
measured vector implementation. The physical results below supply the tested scope; this derivation alone is not device evidence.

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

P3 now adds `fp8_unpack.csl`: nine synchronous vector bitwise/shift/add operations
extract interleaved half values without a scalar table or whole-weight expansion.
`fp8-tile-fast-hw-001` proves all 65,536 packed pairs transform as specified, and
32 original-weight 2x128 cases retain exact scalar FP32 outputs after rescaling.
See MILESTONES.md for scope, independent bounds, physical timings and release.
All transformations retain the pinned original weights; no recalibration occurs.

The measured critical path still spends 829 of 1,286 cycles decoding weight and
operand bytes. This motivates predecoding only the next small resident weight
tile, and multicasting an already converted operand when appropriate. Such work
can move off the dependency path only after buffer/event ownership and overlap
are implemented and measured. No overlap is inferred from these synchronous tests.

## Direct encoder and full group convention

P6 now physically qualifies the complementary FP32-to-E4M3FN direction. Normal
finite values use IEEE exponent/mantissa rounding with an explicit ties-to-even
bit. Subnormal FP8 values use an exact fixed-grid FP32 addition before extracting
the rounded grid index; signed zero and finite saturation follow the original
codec. All53,725 frozen direct cases match the independent oracles. This is scalar
CSL bit manipulation, not yet a vector encoding instruction.

`fp8_quantize.csl` preserves the original group128 maximum, lower clamp1e-10,
scale multiplication by1/448 and per-value division. All44 frozen physical group
cases give identical FP8 bytes and scale bits to the original kernel and PyTorch.
The approximately2x group improvement still leaves a roughly22,800-cycle serial
critical path. A spatial producer must reduce the maximum, distribute the same
scale, encode partitioned values and assemble its operand packet on device before
it can replace P5's already-quantized input fixture.
