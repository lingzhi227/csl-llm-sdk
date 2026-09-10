# Gated RMS128 and recurrent-output composition

The selected BF16 profile follows the pinned official
`Qwen3_5RMSNormGated.forward` body at Transformers commit
`4815a0a6a064214f2d8208c094464a5a6b76ca8d`, lines225–234. The input is
converted to FP32 for RMS normalization; the normalized vector is converted to
the original BF16 dtype before multiplication by direct BF16 gain. That product
has BF16 dtype. Multiplication by SiLU of the FP32 gate promotes to FP32, followed
by the final BF16 conversion. Gain is w, and activation is z*sigmoid(z).

The recurrent fallback returns its original Q dtype. At the composed BF16
boundary, the consumer therefore RNE-casts the received FP32 recurrent output to
BF16 before normalization. Recurrent state remains FP32. Normalization never
passes through host arithmetic in the candidate path.

## Numerical observations and gates

The kernel records normalized FP32 values, normalized BF16 bits, FP32 gain
products, BF16 gain products, exp(-abs(z)), SiLU, final FP32 products and final
BF16 bits. Every BF16 conversion is checked exactly against actual preceding
FP32 bits using an independent nearest-neighbor oracle. Sixteen signed tie,
neighbor and exponent-carry probes exercise the conversion in each standalone
call. Mathematical-oracle BF16 parity is reported separately and is not used
as the conversion gate.

RMS uses mean128 and epsilon1e-6. The FP64 reference propagates gamma256 sum
error through the denominator, sqrt, inverse and normalized multiplication.
Separate sqrt and inverse relative budgets are2^-20 against their actual
arguments. exp(-abs(z)) has a2^-20 relative budget; SiLU has a2^-18 budget against
independent FP64 evaluation. Gain and final products are separately checked
against the actual preceding operands. These thresholds are frozen before
execution; zero annihilation is checked explicitly.

The finite fixture domain is BF16 x/gain within[-2,2] and BF16 z within[-16,16].
Stable SiLU uses exp(-abs(z)), with the negative branch e/(1+e) and positive
branch1/(1+e). Overflow and subnormal correctness are outside this qualification.
The SDK [math library](https://cerebras-sdk-docs-130.netlify.app/csl/language/libraries)
provides the suffixed exp/sqrt functions; actual installed SDK2.10.1 behavior is
checked by the gates rather than assumed from documentation.

Four warm standalone calls include zero input, gain0, negative/zero/positive
gates, large gate magnitudes and a late-cast counterexample. Alternative
offset-gain, sigmoid-only and late-cast results are computed as adversaries.
Counts are BF16 bit differences and may include signed-zero differences; the
nonzero fixtures distinguish the mathematical operations as well.

The CPU reference extracts the unchanged official class/method bodies, removes
only the optional hub-dispatch decorator, and supplies explicit torch/nn/ACT2FN.
BF16 parameters and actual intermediate dtype promotions are verified. A
separate independent mathematical-output comparison allows BF16 quantization
differences; it does not substitute for the device's exact conversion checks.

## Three-PE ownership and readiness

PE0/PE1 retain the qualified64×128 FP32 state shards. PE2 has only small
normalization buffers. After output reduction, PE0 retains global y in its
guarded output and reuses its dead send body for131 words: generation, token,
phase4 and128 raw FP32 output words. Color4 crosses PE1 statically to PE2. The
consumer records the full packet, converts input, computes all gated RMS stages
and writes output before sending a three-word phase5 completion ACK on color5.
PE0 validates the ACK before committing/unblocking its token command.

An explicit host `arm_consumer` command returns only after PE2 posts the receive;
PE0/PE1 perform no arithmetic during this command. The host then launches the
token command on all PEs. PE2 joins token-command arrival and ACK-send completion,
supporting either ordering, and unblocks its token command exactly once after
both. The logic allows either arrival order; only the observed successful order
is device-qualified, not both permutations. At most one fabric operation is in flight on each PE. Every source remains
immutable until its send callback; DSR5/UT2 retain their transfer leases.

Color4/5 routes on PE1 persist across token completion and request reset. PE1's
earlier local unblock does not release its transit routes. PE0's receive buffer
eventually contains a phase5 ACK header plus the retained earlier phase3 body;
the driver checks these separately, never as one coherent frame. The peer's
phase2 receive body still supplies the global delta observation.

All four recurrence tokens and three request generations are reused. Complete
state, prediction, delta and output are rechecked independently alongside
consumer input conversion, norm stages, packet bodies, readiness events,
commits, input immutability and guards. This is a successful serialized command
protocol, not a distributed transaction or qualified error-recovery mechanism.

Both candidates retain the same48 KiB per-PE admission ceiling including4 KiB
stack allowance. Standalone simulation is limited to180 seconds; composition
to300 seconds. Compilation is limited to300 seconds. The existing one-heavy-job,
20 GiB RAM, zero-swap, RAM/disk reserves and cache limits remain in force.
