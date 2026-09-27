# WSE-native lowering boundaries

The prior functional program holds the original model on 870,000 PEs, but routes
work through a serial global interpreter. The captured execution is approximately
30.08 seconds per token. The parked fast interpreter is unqualified and bound to
older sources; it is not a new validated baseline.

The new implementation separates five inspectable, serializable levels:

1. **Model semantics.** Original checkpoint revision, all operators, precision and
   rounding boundaries, recurrent/KV state and token feedback. Import the existing
   `configs/model-graph.json`; never derive semantics from a performance paper.
2. **Regions and ownership.** Adjacent layer/operator regions, matrix partitions,
   distributed norm/quantization/head reduction, explicit tensor owners and
   retained state. Manual knobs: region rectangles, row/K partition, replication,
   compatible fusion and precision-preserving weight transforms.
3. **Streams and events.** Producer/consumer ports, payload length, epoch, initial
   readiness, credit return and completion. Arrival activates work; a local
   region controller is not a full-wafer serial RPC. Manual knobs: buffer count,
   chunk size, reduction tree, pipeline depth and tile ordering.
4. **PE resources.** Per-PE routes, colors, queues, DSRs, microthreads, local tasks,
   memory lifetimes and scratch reuse. Reject conflicting simultaneous routes,
   unowned completion signals, missing consumers, and estimated SRAM overflow.
   Final admission uses compiled ELF sections plus a declared stack allowance.
5. **CSL and evidence.** Deterministic generated code, provenance hash, physical
   artifact identity, same-PE timestamps and invocation-correlated release.

The checked stream/PE/CSL multicast pattern, model dependency importer and
compact matrix-bank ownership planner are implemented. P12 adds a complete
physical ownership/lifetime atlas (see COMPLETE-MODEL-ATLAS.md). Full-model routes,
actual composed SRAM admission and a complete executable schedule remain
unimplemented. A verifier for this restricted pattern is not a general proof of
arbitrary graph deadlock freedom. The model and region compiler will be added
incrementally; placeholder stages must not be reported as implemented.

## First event protocol

A root sends a contiguous packet down column zero and across every row through
hardware multicast. Intermediate routers forward without a software relay task.
Every endpoint consumes one packet, records its epoch, and contributes its unique
PE ID plus the epoch to a reduction. A node waits for its own receive and each
child acknowledgement before sending one sum to its parent. The root starts the
next epoch only after the complete tree acknowledges. Input/output queues are
separate; alternating reduction colors prevent RAMP route overlap. A send buffer
is not reused until its send completion callback. Repeated launches reset state.

This measures a bounded global control roundtrip, not an efficient final model
schedule. The eventual inference graph should communicate between adjacent
regions and avoid a full-wafer barrier at each operation.

## Design references and limits

- [WaferLLM, OSDI 2025](https://www.usenix.org/system/files/osdi25-he.pdf):
  topology-aware partitioning and K-tree reduction. Evaluation uses WSE-2;
  CodeLLaMA-34B and Qwen2-72B include subsets of layers. Current upstream has a
  WSE-3 migration, so code-version and hardware-version comparisons are distinct.
- [SPADA](https://github.com/spcl/spada): explicit placement, asynchronous streams
  and multiple lowering levels. Its published stencil scaling is not a Qwen
  inference rate. The local source archive already contains earlier qualification
  work; no large dependency or repository is duplicated here.
- Existing standalone matrix code in `../experiments/fp8_matrix` already exploits
  column multicast. Its host roundtrip alone does not identify decode, local
  arithmetic or reduction costs. Add same-PE timing before attributing speedups.

A 2,000 tokens/s target gives 500 microseconds per dependent token. It is a budget,
not a prediction. All 64 layers, embedding, full head, state updates, selection,
feedback and observation must be accounted for. The old baseline assigns each layer a small exclusive region; its aggregate
wafer bandwidth cannot be applied to one layer at once. The new bank plan below
shares broad spatial compute regions across sequential layers. Scalar FP8 expansion and nearly full local SRAM are additional
constraints requiring measured kernel work.

## Temporal weight banks across broad spatial compute regions

Layer regions are logical ownership boundaries, not a requirement to give each
layer a small permanent exclusive rectangle. The native mixed-precision simulator
probe completed 254-element FP16/FP32 FMA vectors in about 135 cycles including its
loop (subsequently physically qualified in P2). A baseline 272x128 weight tile thus
contains far too much sequential work for a roughly two-microsecond projection
budget, even before scalar FP8 expansion or communication.

The next placement candidate therefore distributes small tiles of **each**
projection across many more PEs, and packs tiles from **different layers** into
each PE's resident compressed weight bank. For example, a 2x128 tile of the
17408x5120 gate matrix needs 348,160 participants with only 256 original FP8 bytes
per participating PE. This is an analytical partition count, not an admitted
layout. It must still include scales, bank descriptors, code, stack, state and
routing, and account for phases with multiple tiles per PE.

The bank exposes its next local tile early: decode into bounded scratch while
other regions execute, receive operands asynchronously, consume the decoded tile,
then release its buffer. A PE that is inactive in the current projection can
prepare its next projection. Explicit buffer ownership, prefetch readiness and
completion credits are required; a throughput claim cannot assume overlap that
has not been measured. This preserves weight/compute locality without expanding
the entire FP8 model into a BF16 copy that would exceed single-wafer SRAM.

Manual controls will include small-tile dimensions, tensor-to-bank phase offsets,
regional reduction geometry, prefetch distance and one/two scratch buffers. Compare
this candidate with dedicated layer rectangles using measured critical paths and
link occupancy. No layout is selected merely from wafer-wide peak bandwidth.

`spatial/banks.py` now represents the compact ownership candidate. FP8 and BF16
have separate cyclic streams and explicit phases; first-use semantic order binds
all 498 matrices. A tile's owner and local offset are computed from prefix counts,
without a hundred-million-entry materialized table. Each row=2 FP8 tile uses 256
original bytes plus four bytes holding an exact expansion of its original BF16
scale. Each BF16 tile uses 512 original bytes. All original tensor coverage and
aggregate payload conservation are checked.

The current 853,616-bank plan produces only four occupancy classes: (111 FP8,
11 BF16), (112,11), (111,12) and (112,12). With declared code/stack/scratch/descriptor
allowances they use 46,756 to 47,536 bytes. This leaves little space for unmodeled
routing logic. The 16,384 reserved actor PEs are a capacity reservation, not proof
that recurrent/KV state and other operators have legal placements. Physical bank
coordinates, actual executable SRAM and routing remain explicit admission gates.

The first maximum-occupancy component compile (`fp8-bank-sim-001`) fits at
46,768 bytes including the declared stack. This executable includes dynamic FP8
slot addressing and the single decoded-buffer readiness lease. It does not yet
include BF16 execution, descriptor dispatch or communication, so its 1,360-byte
margin cannot be treated as their proven budget. Adding those paths must repeat
compiled admission; if necessary the placement/packing policy must change before
launch, without increasing the 48,128-byte ceiling.

P5 adds a checked regional computation backend: column operand multicast,
arrival/decode readiness, native ordered FP8 dots, ascending-K row sums and returned
completion. Same input and output contracts allow a serialized/overlapped schedule
comparison without changing arithmetic. Its source/resource details and measured
limits are in REGIONAL-GEMV.md. Full bank placement and dynamic activation scale
production remain separate compiler boundaries, not implicitly supplied by the
host fixture used for component qualification.

P7 adds a role-aware composed bank plan (`spatial/banked_region.py`). Root and
boundary roles have explicitly smaller local tile capacities, while ordinary
compute roles retain112 FP8 tiles. All retain12 BF16-sized slots in this component.
The full model's1,251 tensors are not assigned to these role capacities yet. A
future allocator must preserve broad matrix parallelism and short routes as well
as total byte capacity; merely packing the remaining tiles into fewer active PEs
would undo the latency objective. The six-role component passes actual compiled
SRAM after removing unused profiling, but its48-byte minimum margin rules out
assuming additional scheduler/BF16 paths fit without another admission step.


P8 adds a spatial group128 producer (`spatial/quant.py`) with explicit maximum,
scale and payload planes. Its32-PE physical executable directly feeds the native
dot packet contract with no intermediate host transfer. Per-group receive arming
still uses a host readiness barrier, and a single constant weight tile is prepared
once. Full model lowering must place upstream values into these owners and replace
that inter-group host lifecycle with device readiness/credit events. P8 is not an
admission of producer code into the nearly full P7 bank PE.

P9 lowers complete K dimensions into static preorder contraction trees with
explicit parent/child intervals, per-level colors, two receive queues and fixed
addition order. Logical ranks map onto manually selected line or serpentine2D
geometries; the qualified8x28 version packs all224 participants without holes.
The express paths use routers through intermediate PEs and execute only at tree
nodes. It qualifies two original output rows for each complete K width, rather
than supplying full matrix placement. BF16 execution and shared resident-bank
capacities are the next integration gap; see BF16-BANK-NEXT.md. An independent
guarded-quantization proposal is documented separately and is unimplemented.

P10 composes real FP8 and BF16 execution,112/12 resident slots and the tree protocol
at47,472 bytes including stack on physical WSE-3. DSR7 streams lossless BF16
high-half expansion through two FP32 temporaries; DSR4 is shared sequentially by
the native dots. A checked4-byte per-PE type/slot control replaces unused per-tile
metadata. This is representative sixPE executable admission, not the full-model
address schedule. The next boundary is device input arrival and safe repeated
epoch ownership without a host barrier on every contraction; see MIXED-BANK.md.

P11 replaces the representative bank's per-epoch host launches with regional
device control. Request arrival activates arithmetic; tagged child reductions
and explicit send-completion credits govern reuse. The resource IR now separately
assigns microthreads as well as queues, DSRs and local tasks, and rejects collisions.
Two96-epoch physical loops pass while retaining full112/12 banks, at47,792 bytes
including validation instrumentation/stack. A dedicated controller receives every
root result before issuing the next request; this restricted pattern is not yet
a whole-wafer model interpreter or arbitrary graph scheduler. Independent operand
fixtures do not establish neural feedback. See RESIDENT-EPOCHS.md.
