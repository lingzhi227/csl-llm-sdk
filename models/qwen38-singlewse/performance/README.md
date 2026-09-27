# Qwen3.8 WSE-3 spatial performance work

This is the performance successor to the immutable functional baseline at
`6c2f4f5685478ee100f167e42a9f7f22f57dca35`. The baseline implementation, successful
sentence captures and failed strict CPU comparisons are retained unchanged.

The acceptance target is **one physical WSE-3, the complete pinned original
Qwen3.8-27B-FP8 text model, correct dependent sentence generation, at least
2,000 output tokens/s per request**. Batched throughput, simulated time,
partial layers, projected operator rates and commercial serving claims cannot
satisfy that target. The full-model speed target remains unmet; qualified component milestones are recorded below.

P12 now provides an independently audited complete physical ownership and value
lifetime atlas: all1251 tensors,498 matrices,64 layers and105,052,160 real matrix
tiles are covered. Co-resident FP8/GDN state, distributed KV and value actors have
explicit locations and data budgets. This is metadata, not compiled routes/SRAM
or model execution. See [COMPLETE-MODEL-ATLAS.md](docs/COMPLETE-MODEL-ATLAS.md).

The first executable slice is a compiler-generated 2D multicast / acknowledgement
microbenchmark, now qualified on 256 physical WSE-3 PEs. It establishes explicit routes, queues, event ownership and
same-PE cycle timing before adding model computation. It is not model inference.

Run the source checks with `python3 -m unittest discover -s performance/tests`.
Run `python3 performance/tools/build_mesh.py --output <new-directory>` to lower
a checked mesh plan into CSL. See `docs/MILESTONES.md`, `docs/ARCHITECTURE.md` and `docs/MEASUREMENT.md`.

P3 qualifies packed-vector FP8 decoding and original-weight 2x128 native dot
products on real WSE-3: 1,286 cycles versus 11,316 for the unchanged scalar tile,
with exact matched outputs for the frozen cases. This is a local operator speedup;
complete model performance is still unqualified. `spatial/banks.py` supplies a
compact all-matrix ownership/storage candidate with explicit unproven placement
and SRAM gates. Run `performance/tools/plan_banks.py --output <fresh-json>` to
reproduce the estimate without loading checkpoint payloads.

P4 compiles and simulates the largest candidate resident bank at 46,768 bytes
including its declared stack; all112 FP8 slots and retained buffers pass. Prefetch
and compute are measured separately. There is no network overlap or whole-model
SRAM admission yet; see MILESTONES.md for the unimplemented paths and timing scope.

P5 measures a complete8x8 regional dataflow component on physical WSE-3:2,599 to
2,188 cycles with matched overlapping scheduling (15.8% lower latency), including
encoding, multicast, local dots, ordered sums and completion. All four paired
original-weight fixtures pass exactly. This is a partial projection component;
full-model2,000 tokens/s remains unmet. See `docs/REGIONAL-GEMV.md` and the frozen
`regional-gemv-hw-001` evidence for what is and is not included.

P6 qualifies the direct encoder on53,725 physical inputs and complete dynamic
quantization on44 groups, with identical bytes/scales and about2.045x reduction
in summed group cycles. P7 composes resident FP8 banks and regional communication
in the simulator:24 epochs and full storage readback pass, with48,080 bytes maximum
SRAM including stack. Its48-byte margin leaves full scheduler/BF16 integration
unqualified. Both results retain their exact tested scope; the original complete
model and2,000 tokens/s acceptance target are unchanged.

P8 qualifies a32-PE spatial group128 producer followed directly by one original
2x128 FP8 consumer on physical WSE-3. All44 frozen groups pass exact code/scale,
subtree-packet and independent arithmetic checks. Producer median2,484 cycles,
combined median3,003 cycles; host arming/readiness and one-time weight predecode
are separate. Both jobs completed normally and released. See
`docs/SPATIAL-QUANTIZATION.md`; this is a component, not complete model throughput.

P9 qualifies complete40/48/136-block K contractions on224 physical PEs in an8x28
mesh, using original projection weights. A static binary tree reduces the longest
case from14,389 to2,274–2,275 cycles (about6.33x for this component). All four cases
and two separately checked summation orders pass, including intermediate results
and retention. Only two output rows per matrix are executed. The failed initial
simulator and artifact-loading attempt are preserved; the successful run reuses
the compiled artifact and releases normally. See `docs/FULL-K-CONTRACTION.md`.

P10 qualifies actual BF16 computation beside FP8 in full112/12 resident banks on
six physical PEs. All12 switching/replay epochs and complete bank retention pass;
compiled footprint47,472 bytes including stack leaves656 bytes. The explicit
4-byte typed-slot control replaces unused per-tile metadata. Root component cycles
are1,252 FP8 and1,213 BF16, with different original operands; this is not a precision
speedup or model rate. See `docs/MIXED-BANK.md`. Device-controlled epochs, full-model
physical ownership/routes and the2,000 tokens/s target remain unfinished.

P11 runs two96-epoch resident sequences entirely under device control on physical
WSE-3. Request arrival triggers arithmetic; tagged reductions and send-completion
credits advance the loop. All4,608 local/subtree checks and full retention pass,
with47,792-byte maximum SRAM including stack. Complete loops average about1,765
controller cycles/epoch, including multicast, compute, reduction, return and gaps.
Neural inputs are independent preloaded fixtures; this is not autoregressive model
throughput. See `docs/RESIDENT-EPOCHS.md`. Full-model ownership/routes and2,000
dependent tokens/s remain unfinished.


P13 qualifies the complete128x128 FP32 recurrent state on16 physical cohostPE,
each retaining111 original FP8 tiles.96 dependent updates plus8 reset replay,
13,312 output values, complete terminal states, native FP8 dot checks and full
weight retention pass. Physical component latency is6,755–6,758cycles; actual
maxSRAM46,784bytes includes4KiB stack. Both jobs release normally. This does not
include preprocessing or full-model/token feedback; see
[COHOST-RECURRENCE.md](docs/COHOST-RECURRENCE.md).

P14 lowers all 498 matrices to 306 shared-input bundles and 980 events, adding
18 memory completion edges to protect every reused arena range. An independent
audit checks all 44,766,384 entries of four complete abstract reduction forests.
A separate physical 20-PE experiment passes 72 autonomous route/color/queue
epochs with all 705,120 bytes of synthetic bank storage retained. Same-endpoint
dependent send/return latency is 1,132–1,141 cycles; this is not model throughput.
The component's 48,112-byte SRAM footprint leaves only 16 bytes, so combined
neural-kernel admission remains open. See [PROJECTION-LOWERING.md](docs/PROJECTION-LOWERING.md)
and [ROUTE-WORD-QUALIFICATION.md](docs/ROUTE-WORD-QUALIFICATION.md).

P15 adds a complete paired-column input/storage overlay: all 498 original
matrices and 105,052,160 tiles pass independent address/route audits while the
maximum bank payload remains 35,256 bytes. A two-stage hardware counter window
feeds original FP8/BF16 native dots in representative maximum resident banks;
all ten physical window/precision/replay cases and complete original-bank
retention pass, with 47,328-byte SRAM including the declared stack. Both jobs
succeed and release normally.
See [COLUMNAR-INPUT.md](docs/COLUMNAR-INPUT.md) for the tested scope and remaining
whole-matrix integration. The full-model speed target remains unmet.
