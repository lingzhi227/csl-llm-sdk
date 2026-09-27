# Proposed guarded reciprocal quantization (not implemented or qualified)

P8's exact spatial producer still performs an original FP32 division for each
input. A possible next optimization computes one reciprocal per group, multiplies
ordinary inputs, and falls back to the unchanged division near any FP8 rounding
boundary. This is a proposed exact-result optimization, not permission to change
quantization semantics. No speed or numerical qualification follows from this note.

Let s be the original stored FP32 group scale, u=2^-24, r=RN32(1/s),
a=RN32(x*r), and q=RN32(x/s). For the group convention max(abs(x),1e-10)/448,
s and r are finite normal FP32 numbers. For normal quotients and round-to-nearest,
reciprocal and product rounding imply an error of at most (2u+u^2)|x/s| in a;
q adds at most u|x/s|. Thus |a-q| <= (3u+u^2)|x/s|. The implementation must
formalize the conversion from this relative bound to adjacent FP32 encodings;
it must not assume three encoding steps at an exponent boundary without proof.

For |a| >= 2^-6, ordinary E4M3 rounding midpoints lie halfway through each
20-bit discarded-mantissa interval. An implementation can conservatively detect
a small encoding neighborhood around remainder0x80000, then perform the original
x/s there. Values below2^-6 should initially use the original division as well.
Zeros may retain their sign directly; clipped values need an explicitly proved
safe rule. The neighborhood must account for both reciprocal/product error and
the original division's rounding, including exact midpoint ties. A conservative
guard is preferable to silently accepting different FP8 bytes.

Before device changes, prove the guard's domain and coverage independently of
candidate observations. Test both signs, every finite FP8 midpoint neighborhood,
scale exponents at both limits, tiny/subnormal inputs and seeded random IEEE data.
Freeze the guard and fallback rules, then compare device outputs against the
original division and independent PyTorch group quantization. Record actual
fallback counts and whole-producer cycles, not multiply-only timing. Keep P8's
original division implementation and all frozen evidence intact.

If this optimization is admitted, distribute the one reciprocal with the scale
and retain the original scale in the65-word GEMV packet. All PE receive, buffer
and DSR ownership and actual compiled SRAM must be rechecked. Full inference
still requires complete BF16 paths, weight-bank allocation, device-driven epochs
and all64 layers/full vocabulary with real dependent-token measurement.
