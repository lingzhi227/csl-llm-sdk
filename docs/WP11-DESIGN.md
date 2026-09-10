# WP11: original Q/K RMS256 and bounded device partial RoPE64

One query head and one key head, each256dimensions. Inputs are original synthetic
BF16 projections and distinguishable BF16 offset-gain weights in[-2,0.5]. Ordinary
RMS uses FP32 square/mean256+epsilon1e-6, sqrt/inverse, normalization, then1+weight
and one BF16 cast. Both zero and negative gains are present. There is no projection,
all-head GQA, attention/cache composition, full layer or long-context claim.

The pinned default RoPE source produces32immutable FP32 inverse-frequency values
for theta10000000 and rotary dimension64. Integer positions0–7 are read from device
control after RMS; angle multiplication, range reduction and sin/cos run in CSL.
There are no host-generated runtime trigonometric or normalized candidate inputs.
Equal text axes duplicate the32frequencies in split-half order. The first64values
use separate BF16 product0/product1 and BF16 sum; the remaining192values pass through
exactly. Per-operation conversion and signed-zero checks accompany final intervals.
Zero-angle sine/cosine must be structurally exact0/1, separately from error gates.

## Trigonometric gate fixed before device outputs

Use nearest-quadrant selection, then split-pi/2 reduction with high1.5703125 and a
rounded FP32 low constant. The selected quadrant is0–4 for angles[0,7]. High times
quadrant is exact; first subtraction is exact under Sterbenz (or identity at0).
Low multiplication/coefficient/final subtraction error is bounded by4.69749e-8.
Reduced magnitude is conservatively at most0.786. Explicit sin degree13 and cos
degree12 Taylor polynomials use FP32 Horner operations and final quadrant signs.
A forward-error recurrence accounts for rounded coefficients, squared argument,
each multiply/add and Taylor remainder: total sin<=1.68553e-7, cos<=1.87543e-7.
The predeclared absolute gate remains2^-20=9.536743e-7. Reference-angle error is
separately propagated through source intervals; conditional trig checks use the
actual observed FP32 angle. A28027point grid plus adjacent FP32 quadrant boundaries
has max sin/cos error7.49837e-8/7.05837e-8. Eight fixed device probes repeat each call.
This establishes only the bounded profile; large-position reduction is future work.

## Source and observation contract

Pinned ordinary RMS/rotary bodies retain arithmetic. Framework integration
decorators are removed for CPU extraction; the static frequency method is relocated
as a standalone function with the same explicit config argument. An initial CPU
attempt failed before math because the class integration decorator remained; the
fresh corrected attempt passes10calls. Its original oracle functions/policy are
unchanged in the device bundle, which adds conditional actual-stage validation.

Independent FP64 source intervals include RMS approximation, exact rational BF16
endpoint selection, angle/trig perturbation and both separate product casts/sum.
The official CPU result matches all5120nominal output bits. Each head has248–256
single-value output intervals. Some near-zero intervals include many representable
BF16 values while remaining absolutely narrow; maximum numeric span1.5258789e-5.
These bounds are frozen before SDK and are not adapted to device observations.

Ten calls cover positions0–7 then reset0/1, asymmetric Q/K, split-half basis,
nonuniform/zero/negative gain, nonzero tail, and zero after nonzero. Every call reads
all inputs/weights/frequencies, actual RMS/trig/rotation stages and casts, probes,
guards, state/events and output. Position0 numeric identity, tail192bit equality,
all actual BF16 casts, zero-product/sum signs, zero-angle trig and normal stop are
independent gates. Out-of-profile rejection exists in source but is not exercised
by this declared successful-call fixture and is not a tested refusal claim.

## Resources

Fixed185physical copies/49054hostu32slots/134604native payload bytes/12launches.
Maximum SDK host buffer4104bytes. One synchronous PE, no application colors or
asynchronous callbacks. Estimate130seconds based on WP10's112-second242KiB sequence
and this smaller131KiB payload, with additional short transfers and bounded trig;
simulator hard180seconds, compile300seconds. Actual SRAM end+4096stack<=49152 before
simulation. Original single-heavy20GiB/Swap0,8GiB available RAMreserve,20GiB cache and
32GiB diskreserve apply. No install/download/GPU/MPI/hardware/extra agents.
