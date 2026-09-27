# Compact physical projection dataflow

P18 is physically qualified by `compact-projection-hw-002`. All six original
fixture cases, exact replay and complete bank retention pass, with normal stop
and both jobs released. The five complete BF1648x5120 calls each take3898 cycles
versus P17's10824:2.77681x, or63.99% lower latency. This comparison includes
input transfer, native computation, reductions and full output delivery, with
the input-owner change described below. It is not a full-model tokens/s result.

The first shared projection backend maps K partitions across one physical axis
and output-row partitions across the other. For the complete original48x5120
BF16 matrix,960 workers occupy40x24 PEs. Forty input owners sit along the K edge;
one kick starts40 independent65-word input streams. The complete application is
41x25, including the border/controller. Axis transposition produces25x41 from
the same representation and protocol. This is a candidate, not a universal layout.

The arithmetic partition remains2x128. The fixture NPZ is bit-identical to P17,
including all original target weights, other resident banks, all inputs, ordered
FP32 subtree references and independent FP64 bounds. Only coordinate metadata
changes. Input owners preload the same2600 words at40 distributed locations.
The timed interval starts at the controller kick and ends at delivery of all48
outputs; it includes all40 input transfers, native dots, reductions and gather.
P17 instead starts with all input words at its controller. Both exclude prior
host input placement, but the ownership difference must accompany comparisons.
This is deliberate producer-placement optimization, not host neural computation.

The actual software boundaries are:

- `ProjectionSpec` / `ProjectionFlow`: original dimensions, tile partition,
  physical K-axis orientation and explicit structural admission budgets.
- Region/input/output ownership, stream paths/lengths, completion dependencies,
  buffer release conditions and queue/DSR/microthread leases form an inspectable
  document. The initial completion DAG describes completed streams; it does not
  impose additional whole-region barriers on the streaming implementation.
- One lowering emits routes and role parameters into a shared
  `protocols/matrix_epoch.csl`. Input arrival triggers arithmetic, child callbacks
  trigger ordered joins, and send callbacks credit the next gather receive.
  The independently qualified native arithmetic modules remain separate.
- `audit_projection_flow.py` independently traverses emitted routes and validates
  original partition, native/gather recipients, completion dependencies and exact
  directed-link traffic. Actual ELF SRAM remains a separate execution gate.

The first family supports manual dimensions/tile choices, axis orientation and
cost thresholds. Routes/leases/buffer lifetimes are explicit and inspectable;
arbitrary route editing, multiple buffers and alternative tree embeddings are
not yet exposed as qualified interchangeable backends. Do not mistake this
restricted reusable chain for a general tensor/compiler framework.

Static measurements for this candidate are24 input hops maximum,128 transport
hops along the longest single input/native/gather contribution route, and66 words maximum per
directed link per epoch including input teardown. There are1961 directed links,
4727 route entries and1001 streams. Input/control traffic is63360 word-hops,
native reduction7992, ordered gather552, delivery96 and kick40. These are traffic
and structural counts, not predicted cycles or a tokens/s result. This is not the longest complete credit/control critical path: the controller
gather serializes24 chunks, and callback waits/backpressure still require
execution measurement.

The first8x32 orthogonal candidate is rejected before compilation because its
native/gather/kick/return planes would require an unqualified SDK color. The4x64
candidate exceeds the current single-contribution path budget with a single unfurled K row.
Neither conclusion forbids those arithmetic shapes: partitioned K regions,
hierarchical ingress and a different routing/credit schedule may admit them.
The point is to change spatial organization when a tile choice needs it, rather
than force an arithmetic optimum into a poor communication layout.

Extension must be tested against actual major shapes, including17408x5120,
5120x17408 and the complete vocabulary head. Their input/output producers,
consumer locations, multiple bank phases, state placement and capacity cannot
be inferred from this small original matrix. Large outputs must reach actual
nearby consumers instead of growing a controller buffer. Preserve the current
baseline, revise atlas ownership when justified, and keep exact tensor identity
and per-class SRAM proofs. Whole-model correctness and>=2000 dependent tokens/s
remain the acceptance target.

The initial simulator compiles1025 PE placements into109 application images.
Its maximum SRAM including4096-byte stack reserve is48048 bytes, leaving80 bytes.
This is executable memory admission for the measured component, not spare room
for a production scheduler. The corrected physical002 footprint is48112 bytes including the same stack
allowance, leaving16 bytes. The initial and corrected snapshots remain separate.

Simulator `compact-projection-sim-001` passed normally with3687 controller cycles
from kick to complete output, versus9328 in P17's simulator. All1920 local
values,1920 subtree values,48 outputs,62400 input words and122880 target words
pass, including both negative controls/restoration. Maximum original-matrix FP64
error is3.845198079943657e-7, with the same frozen P17 oracle/bounds. Original
background retention and six-case replay remain physical gates. The physical
attempt is separately bounded; no simulator ratio is a physical speed claim.

Physical001 is not accepted. Compilation succeeds; mixed case0 and zero case1
pass at3891 cycles. Tiny case2 has exact local/subtree/counters and final root
chunks, but28/48 returned values mismatch with pair reordering. Runtime exits1
and its job is terminal FAILED/released; no full-bank/replay acceptance. The
initial gather queue binding persisted from the previous epoch's final stage.
Under parallel arrivals, a later-stage child may therefore enter that queue
before the next first-stage receive is configured. This is the current causal
hypothesis, not independently proven from a hardware trace.

The shared protocol now restores the first gather color during arm, before the
all-receiver readiness fence and kick. Simulator002 requires mixed and tiny
nonzero epochs (zero outputs alone cannot expose reordering), with all original
per-epoch numerical and packet checks plus terminal target retention. Acceptance
requires a subsequent complete physical six-case/replay/full-bank run.

The corrected simulator002 passes two distinct nonzero epochs at3687cycles each,
with124800 packet words and122880 target words checked, three negative controls
and restores, and normal stop. Physical002 then passes11520 local values,
11520 subtree values,288 returned values,374400 packet words, all producer
packets and per-PE callback/teardown counters, replay and33832240 original bank
bytes plus13520 zero-padding bytes. The bit-identical original P17 fixture hash
is retained. BF16 calls all measure3898cycles; the matched representative FP8
smoke call measures3694 versus10633cycles, without becoming a full FP8 matrix.

The physical compiler/runtime lifecycle intervals are51.470s and81.451s; they
include scheduling/startup/cleanup and are not kernel time or provider billing.
The0.289666s resident-upload/alias/fence interval begins after runtime readiness
and excludes artifact upload/runtime startup. No tokens/s is inferred from these
component measurements. Artifact SHA256:
`280caec384e70dc908d3c4856b62a2ffe87314964241f22898384185ce73586f`.
All149 owned account jobs are terminal with zero system assignments at the final
resource audit. Both failed and accepted physical attempts remain frozen.

Restoring the initial queue binding fixes the observed six-case/replay regression;
this supports the stale-binding hypothesis. No independent hardware trace is
claimed. Queue state at an epoch boundary is part of the protocol contract, not
implied solely by a dataflow completion DAG. The next real multi-operator graph
must lower such initialization, readiness, credits and buffer lifetimes explicitly.

`mlp-subgraph-plan-002.json` imports real graph nodes13..18: full5120/17408 RMS,
gate/up, SiLU/multiply, down and residual, with explicit shared quantization and
retained residual lifetime. Its267429760 original weight bytes and1044480 native
tiles are semantic/resource counts only. Physical regions, consumer-direct flows,
redistribution, composed SRAM and execution are still unqualified. Do not report
this initial importer as an executable MLP or whole-model compiler.
