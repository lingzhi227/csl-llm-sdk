# Full MLP spatial ownership and fusion candidate

P19 advances the real layer0 MLP boundary from P18's semantic importer. It does
not execute that complete subgraph or report a new physical speed result. The
complete model and >=2000 dependent tokens/s target remain unmet.

`mlp-regions-plan-003.json` binds the original5120/17408 dimensions, original
matrix coordinates and replicated scale indices to a new candidate geometry.
The independent audit covers all1,044,480 native matrix tiles and267,386,880
original FP8 matrix bytes exactly once. Including original scales and the norm
weight, the layer boundary owns267,429,760 original bytes. It does not change the
P15 global atlas or assign resident slots for the whole27B model.

## Spatial organization

There are136 intermediate groups of128 values. Each group contains64 adjacent
gate/up output pairs, two complete K40 contractions per pair, a local activation
column and40 independent down-output collectors. Gate and up roots face their
shared activation actor. Down reuses the2560 gate-worker coordinates in each
group after native sends and forwarding paths have drained.

The default nine-column arrangement occupies747x1080 application coordinates:
696,320 gate/up workers,8,704 activation actors,5,440 down collectors,720 header
distributors and40 retained-input owners. These711,224 active roles are within
an806,760-PE bounding rectangle. Eight columns also fit the geometric constraint;
neither shape has been chosen by a complete physical latency comparison.

Each original input group has one normalized/quantized owner. Forty horizontal
header streams fork into separate gate/up vertical streams through real relay
actors. No router color conversion is assumed. Down has40 output stripes, each
of128 values: region partials reduce vertically, turn into separate header rows,
and reduce horizontally into their original residual owners. No single PE must
collect all5120 down values.

The independent audit traverses702,880 currently lowered stream paths,
3,767,796 PE/color entries and1,811,540 directed links. These comprise the input
distribution, all gate/up contractions, packed adjacent activation operands and
the inter-region down reduction. The maximum traffic for **these families** is
128 words per directed link. Local down broadcast/gather, group quantization,
RMS, readiness and termination traffic are still missing;128 is not the total
MLP link-load bound. Counts and word hops do not predict cycles or tokens/s.

The down expression contains each of136 original K groups exactly once. Its
FP32 addition order is explicitly recorded as vertical and horizontal binary
expressions, with maximum binary depth22. This differs from the semantic
ascending-block order and still requires complete-subgraph numerical checks.

## Fusion follows actual dependencies

`spatial/mlp_fusion.py` records four restricted decisions. They support the
complete-MLP integration; optional further fusion is not a prerequisite for
obtaining an executable, measurable subgraph.

1. RMS output rounding, exact input-group quantization and the gate/up fork
   share each input owner. The original BF16 residual remains live.
2. Projection roots round two FP32 sums into one packed BF16 word each. Their
   adjacent actor consumes the two words, applies SiLU, rounds toBF16, multiplies
   by BF16 up, rounds again, and produces the local maximum contribution.
3. A128-value cooperative group computes its maximum/scale, encodes each pair
   directly into the native FP16/256 packet format, and feeds down. Its readiness
   need not wait for all17408 intermediate values.
4. Final down chunks round toBF16 and add the retained BF16 residual at its owner,
   round again, and feed the real next-layer interface. Next-layer local square
   accumulation may begin on arrival, but its global RMS dependency remains.

`csl/mlp_fused.csl` implements the packed numeric helpers. It calls the unchanged
functional `qwen_math.csl` and existing FP8 encoder, with source provenance. No
BF16 boundary is removed. The helpers currently have a numerical simulator unit
qualification; they have not been integrated into the complete distributed MLP.

## Readiness, queue state and memory

A local native send callback does not prove that all routes through that PE are
idle. A row's two actual contraction-root deliveries provide the data dependency
for a row fence. A worker must see that fence **and** its native send callback,
install the next routes, rebind the first gather color, then acknowledge readiness.
The group joins all64 row readiness acknowledgements before releasing its down
packet. Down chunk buffers retain ownership until their outgoing callbacks;
residual reuse also depends on actual successor consumption.

`region_epoch.py` and its CSL counterpart make those local transition preconditions
explicit. Tests reject missing fences, premature acknowledgements, old queue
colors, stale epochs, duplicate arrivals and premature buffer reuse. The CSL unit
exercises both callback orders across three epochs. It **does not implement or
prove** the distributed fence, asynchronous network liveness or termination tree.
Those remain required work, including the next-token initial route configuration.

Geometry is not SRAM admission. In the old atlas,860,880 of870,000 PEs hold
banks, leaving9,120 dedicated actors. Treating all14,904 new helper roles as
bank-free would exceed that allowance by5,784. Enough helpers must cohost original
weights, or the complete packing must change. One layer also expands its original
scale entries into1,044,480 FP32 tile copies,4,177,920 resident bytes. That cost is
recorded rather than hidden inside the original checkpoint size. No model weights
have been removed, and no code/stack allowance has been reduced to make it fit.

## Numerical evidence and retained failures

`mlp-fusion-sim-001` fails compilation because a parameter used the reserved CSL
type name `color`. Its source, logs and release receipt are retained.

`mlp-fusion-sim-002` compiles but fails its first numerical case at two scale
words, each one ULP from a literal PyTorch division by448. Its other448 output
words match, including BF16 results and FP8 encodings. The device used the
unchanged FP32 reciprocal multiplication from the original implementation; the
new oracle had used literal division. Failed actual arrays were saved remotely
before the assertion, and the diagnostic is published.

`mlp-fusion-sim-003` keeps both independent PyTorch oracles. All1350 output words
across three128-value cases match the existing reciprocal-multiply convention
exactly, including BF16 midpoint cases, zero, signed nonlinear operands,
normalization and residual boundaries. The separately retained literal-division
comparison differs at five scale words over the three cases, at indices
`[256,385]`, `[256,385]`, `[385]`. It is not claimed bit-identical. This verifies
fusion preservation of the existing implementation, not full-model alignment
with every alternative division lowering.

The simulator stops normally. Its one-PE **test program** uses11,744 bytes through
the low-section end plus4,096 bytes reserved stack:15,840 total. It contains no
resident weight bank or distributed routes and therefore does not admit any
production helper/worker SRAM profile. No timing or speedup is claimed.

All three bounded workstation attempts have terminal service and free-lock
receipts. P19 dispatched no ALCF compile or run. The final account audit records
149 terminal owned jobs, no active owned job and no owned system assignment.
Other users' assignments were left untouched; provider billing totals are not
available from the job snapshot.

## Next executable boundary

The remaining work is concrete: bind resident slots and measure actual bank plus
actor code/data/stack; lower local max/scale/gather, down broadcast/gather, RMS,
row readiness and reset; integrate the original full-dimensional projections
with actual successor consumption. Measure the complete subgraph including
redistribution and stalls. Avoid replacing this boundary with an open-ended
series of independent epilogue or matrix benchmarks. Further optional fusion
should be selected by its effect on that complete path.
