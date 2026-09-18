# Full original layer3 MLP acceptance

This is one original Qwen/Qwen3.8-27B layer3 MLP, revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, with dimensions
5120 ->17408 ->5120. It is not a complete transformer layer or model.

The four frozen module-boundary inputs are dense_dyadic, changed_dyadic,
last_column_onehot, and zero_after_nonzero, in that order. The accepted CPU
reference is reused without a forward rerun. Its reference.json SHA256 is
62274b756616094eceff07b27b790da9d74edde997b1300bbac854ce4d380972.
All consumed source and official arrays must match the hashes in that receipt
before a physical runtime is constructed.

## Computation and source policy

Each matrix PE holds128 output rows by96 original BF16 columns. The last
projection and down shards each contain32 original columns and64 zero columns.
The local kernel performs96 forward contractions, holding FP32 partials.
A right-to-left chain adds54 projection or182 down partial vectors in FP32.
There is one BF16 round-to-nearest-even cast at each root. Gate/up words travel
directly to136 nonlinear PEs. Their rounded SiLU product is ordered by group,
broadcast on device, and becomes the down projection input. No neural
intermediate is recomputed or supplied by the host.

Every observed local input is checked against the original hidden or the actual
device BF16 product, including all padding. Every local128-row partial is tested
against a conditional enclosure using the corresponding original weight words.
The vectorized point-input bound uses the already accepted linear_bounds policy:
gamma(96,u64) for the FP64 diagnostic sum, gamma(192,u32) for local contraction
rounding, an outward magnitude bound, and the accepted FTZ allowance.1152
synthetic rows gave bit-identical bounds to that accepted scalar implementation.
The local checker is conditional; it cannot replace the independent source gate.

The retained full source intervals remain a conservative independent gate only
under the following explicitly tested domain. Each actual BF16 product is normal
and exactly representable in FP32, or zero. Observed partials and computed chain
intermediates must be normal FP32 or zero. Thus each output has at most n local
effective rounding sites plus ceil(n/96)-1 chain sites, at most2n. The existing
full source gamma(2n,u32) and2n FTZ allowance conservatively dominate those sites.
Subnormal ambiguity, overflow, non-finite values, or domain failure rejects this
candidate rather than silently widening a bound.

The actual captured FP32 partials are separately added in exactly the device's
right-to-left order with native FP32 additions. Every multicast receiver must
match that result bit for bit. All root casts, all gate/up-to-nonlinear words,
all product-to-down inputs, and the complete5120-word output handoff are checked.
All17408 nonlinear rows retain the accepted actual-operand exponential, sigmoid,
SiLU, product and BF16-cast checks, plus the unchanged source enclosures.

The original official BF16 outputs are compared and mismatch counts reported.
Bitwise parity with the official CPU implementation is not the acceptance rule
for a changed FP32 reduction order. Passing the stated mathematical contract is
not a full-model output-token guarantee.

## State, weights, and release

All57776 application PEs must show exact prepared/completed epochs and cumulative
GEMV, reduction, nonlinear and packet counts for each input. The fourth input
must clear every earlier matrix value and produce zero output. Extra diagnostic
arrays on the136 odd idle PEs must remain all-zero. Every original weight and
padding word is read back after all four inputs; all86 native-u16 rectangle
hashes must match the original prepared images.

The physical phase has253 copies and9 launches. It performs bounded transport,
state, storage and identity checks, then exits the runtime. Owner-correlated
watchdog evidence must confirm a terminal successful job and no remaining
system assignment. Only then may the separate bounded host auditor perform
the expensive original-weight mathematical checks. Capture completion alone
does not constitute numerical acceptance.
