# WP13: original selected attention-head projections

Accepted scope is original layer3/head0 Q256, rawgate256, K256 and V256 over all5120
hidden-input columns, with four common synthetic BF16 inputs. Eight independent
single-PE four-call runtimes cover1024 selected rows. This is a projection milestone;
complete attention integration, all outputs in one runtime and model inference remain
future work. See the [report](WP13-REPORT.md), [reproduction guide](../examples/wp13/README.md)
and [evidence index](../evidence/wp13.json).

## Original source and input mapping

Checkpoint `Qwen/Qwen3.8-27B` is pinned at
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; tools and evidence enforce this revision.

The qualified Transformers source revision is
`4815a0a6a064214f2d8208c094464a5a6b76ca8d`, with source SHA256
`762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b`.
The pinned projection/view/chunk prefix is executed on CPU with selected-row BF16
Linear modules. Head0 q_proj rows0:512 split into Q256 then rawgate256 within that
head; k_proj/v_proj rows0:256 provide the shared KV head. Attention bias is false.
The full projection matrices and whole model are not instantiated.

The [acquisition plan](../examples/wp13/acquisition-plan.json) preserves verified
metadata hashes, tensor shapes, source offsets and the12 exact HTTP206 ranges.
Three coalesced files contain10,486,784bytes:10,485,760bytes of BF16 projections and
1024bytes of actual Q/K norm offsets. A16MiB aggregate cap and1MiB request chunks
prevent full-shard fallback or unbounded downloads. Range access is pinned and
strict; a whole-shard hash is not claimed to have been independently computed.
Original weight bytes remain outside the source release.

The four complete5120-element inputs are deterministic dense dyadic, changed dyadic,
one-hot at column5119 and zero following nonzero. Dense scale2^-12 follows the
measured maximum selected-row L1 norm94.63631564:16*2^-12*L1<0.5. Original trained
weights are unchanged. [Input words and hashes](../evidence/wp13-inputs.json) and the
[reference recipe](../examples/wp13/reference-recipe.json) preserve exact values.
All four projections use the same original hidden input.

## Device arithmetic and source uncertainty

The accepted persistent GEMV kernel is byte-identical to WP03. Each128-row slab
executes increasing columns0..5119, one persistent FP32 FMA per column:45 tiles112
and final tile80. Only begin clears accumulators. Finalize converts each FP32 result
to BF16 RNE once. Parallel output rows do not change this reduction order; parallel
column reductions would require a different numerical policy.

Independent original row-major FP64 dots and absolute-product sums define the source
interval. Bounds include gamma5120 FP32 accumulation, FP64 reference rounding and
an additive subnormal allowance, then exact monotone BF16 endpoint selection. The
one-hot and zero cases have exact structural radius0. Original weight/input lattices
also establish no subnormal serial-FMA intermediates for these frozen fixtures;
this is not a general hardware FTZ guarantee. Observed bitwise agreement does not
replace the conservative source interval or imply agreement for arbitrary backends.

Dense intervals have maximum numeric width0.0001220703125 and no singleton elements;
radius-vector L2 norm is about1.64–1.66% of nominal-output L2 norm. Large BF16 span
counts occur near zero. Both dense cases reject an all-zero output at1013/1024
coordinates; the explicit permutation candidate[i]=nominal[(i+1)%1024] is rejected
at1015/1016 coordinates. Exact one-hot output rejects both at every coordinate.
These are informative projection gates, not tight end-to-end attention guarantees.

## Storage, transfers and ownership

Global slabs0/1 are Q,2/3 rawgate,4/5 K and6/7 V. Every full-coverage run owns one
128-row slab and all four cases in one loaded runtime. Release precedes the next
begin, so changed input and zero after nonzero exercise reused state. Finalized
FP32/BF16 outputs remain intact until explicit release. Host code only packs original
weights, supplies original input and schedules/checks operations; no candidate dot
or cast is computed on host and uploaded.

Each tile uses guarded column-major BF16 weights, guarded FP32 input and control.
The last32 invalid columns retain nonzero poison (.25 in BF16 weights and7.0 in
FP32 inputs). Prefix readbacks after112 and5040 columns check original-source FP32
bounds and complete state. Every call then checks final FP32/BF16/source/RNE,
guards, original input, fresh weight-guard snapshot,46 timing pairs and release state.
The final call reads the entire resident weight tile. Historical tile immutability
is not inferred from that final snapshot. Begin-before-release rejection is not a
device-exercised malformed-command test, and no future consumer transport is claimed.

All physical copies have width1 and native16 occupies one u32 host container per
BF16 value. Every full batch has603copies,196launches,10,714,296host bytes and
5,405,728native bytes; max host buffer57352bytes. Symbolic sequence and durable
entry/exit records preserve operation order, identity, dtype/shape, byte counts and
buffer hashes. Each record is flushed/fsynced. Exceptions attempt stop while
preserving the first error. Passing requires normal stop and unchanged artifacts.

Declared per-PE arrays total31080bytes, including the kernel scratch. Admission uses
actual ordinary ELF maximum end+4096stack<=49152, not the declared-size sum. SDK
compile/simulation limits are300seconds each; full-batch estimate240seconds, one
heavy job total,20GiB/Swap0, RAMreserve8GiB, cache cap20GiB, free-disk reserve32GiB,
compiler parallelism1 and simulator thread1. No GPU or physical-wafer result is claimed.

## Why the attention composition needs a separate proof

Real K norm offsets include3 values above the old0.5 upper bound, reaching0.75390625.
All nonzero projected Q/K cases fail the older exact-square fixture proof. Their
unrotated normalized tails already exceed the prior component bound3. Projection
uncertainty must propagate through nonexact RMS, offset gain, rotary stages, current
Q, persistent K/V and gate before qualifying original-hidden attention. Existing
standalone and composition policies remain unchanged by this projection milestone.
