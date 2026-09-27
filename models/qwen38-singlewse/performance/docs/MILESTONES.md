# Performance milestones

## P0: checked graph and executable stream foundation

Status: source and simulator milestone. The full-model 2,000 tokens/s target has
not been reached. Physical transport qualification completed separately in P1 below.

Implemented and checked:

- All 1,172 original semantic operators, 1,251 text tensors and 64 layers imported into
  an explicit dependency/region representation; intra-token KV ordering and
 113 delayed state/token-feedback edges are explicit. Physical region placement
  is unassigned, rather than filled with hypothetical coordinates.
- Deterministic stream-to-PE-to-CSL lowering for a rooted 2D multicast and
  acknowledgement tree; manual geometry, packet, color, queue and task controls.
  The restricted verifier checks route continuity, resource collisions, coverage
  and estimated SRAM. This is not a generic arbitrary-graph deadlock proof.
- Seven source tests pass. Complete 3x4 simulator suites 006/008 pass all 5 cases,
  including changed inputs, restarted requests,33/256-word payloads and 128 rounds.
  All endpoint packets, epoch counts, child counts and independent root sums pass.
- 16x16 simulator 007 compiled all 256 PEs and passed its first 2 cases, including 64 rounds
  of 33 words at 2,176.03125 root cycles/roundtrip. It was stopped when observed simulator
  cost showed the complete suite would exceed its 240-second budget. This is a
  partial simulator result, not a full physical or full-model performance result.
- An independent IEEE encoding proof covers 254 finite FP8 codes and 64,516 products.
  It identifies a possible table-free mixed-precision kernel, with device
  subnormal behavior and accumulation correctness still unqualified.

Failed attempts are preserved. Simulator 001 rejected non-symbol export operands;
002 hit the bounded process/thread limit before actual compilation; 003 timed out;
004 exposed a device assertion, and 005 captured the actual checker error: epoch 1
returned the correct sum 90 while its multiply-based check evaluated to 65,610.
006 uses an incrementally maintained device check while the host independently
retains the closed-form formula. No numerical tolerance was relaxed. The cause
of the original integer expression requires a separate compiler/arithmetic probe;
it is not presented as a proven general compiler defect.

For the initial local compile, explicit process CPU affinity corrected LLVM's
thread creation failure without raising the 64-task limit. Simulator fatal-log
monitoring now terminates a halted simulator promptly. Hard memory/no-swap,
CPU/runtime and shared-lock limits remain in force.

## P1: physical 256 PE stream qualification

`mesh-transport-hw-001` completed all five physical cases on one WSE-3. Every
endpoint packet, epoch count, child acknowledgement count and independent root
checksum passed, with changed inputs and restarted requests. The 33-word cases
measured 2,248 root cycles per roundtrip, and the 256-word cases measured 2,500.
These are complete multicast plus all-endpoint acknowledgement cycles, not model
token latency. Clock frequency is not assumed. Host completion plus audit transfer
is reported separately in the raw compact results.

Both jobs succeeded and released normally, with no cleanup errors:
`wsjob-hd2dkp3emcbrcfmv9ytx2d` (compile) and
`wsjob-nwki8fjpbhdyphj3rs2kfw` (execute). Stage wall times were 232.454 seconds and
71.407 seconds, including infrastructure work. They are not provider-billed node
hours. A fresh account/system audit found no owned active jobs or assignments.

All 256 compiled application ELFs passed the local SRAM gate; maximum low-section
end plus the declared 4,096-byte stack allowance is 10,752 bytes, below 48,128.

## P2: exact native mixed-precision products on WSE-3

`fp8-native-hw-001` passed all 64,516 ordered products of finite FP8 encodings,
including encodings that become FP16 subnormals, plus two 128-repeat accumulation
cases and weight retention. Actual FP32 outputs equal the independent expected
finite values exactly, with zero signs canonicalized by the positive-zero FMA
accumulator. Original tensor weights, packed extraction and complete GEMV are not
yet part of this probe.

A 254-element native mixed FP16/FP32 FMA vector takes 135.3125 to 135.3203125 cycles
per iteration in this loop. This is an operator measurement, not a model token
rate. These results motivate smaller per-PE matrix tiles and interleaved weight
banks, rather than assuming a large resident tile can meet the per-stage budget.

Compile `wsjob-6knikeasu8pkw2bv6p3gzg` and execute
`wsjob-btccu78jtevwtjzupy3y5h` both succeeded and released normally. Their stage wall
times were 41.464 seconds and 71.394 seconds. A fresh resource audit confirms no
owned active hardware jobs or system assignments.

The compiled one-PE program uses 10,128 bytes including the declared
4,096-byte stack allowance, below the 48,128-byte application ceiling.

## P3: physical packed FP8 original-weight tiles and bank ownership plan

`fp8-tile-fast-hw-001` passes all 65,536 packed two-byte patterns through the
vector bit insertion transform, and 32 dot cases built from eight original
layer-0 matrix tiles and four fixed activation patterns. NaN encodings participate
only in the bit-transform test; numerical fixtures contain finite original FP8.
The original shard and fixture are hash pinned. Both PEs receive identical inputs:
one executes unchanged scalar FP8 arithmetic, the other vector decoding followed
by native mixed FP16/FP32 arithmetic in the original 128-term order.

All optimized outputs are bit-identical to the scalar outputs (zero signs
canonicalized), finite, and within the frozen independent FP64 absolute-product
bound. The maximum observed FP64 error is 3.814697265625e-6. All weights, scales,
packets and call counts pass retention checks; normal stop completes.

Physical 2x128 tile totals are **11,316 scalar cycles versus 1,286 vector/native
cycles**, an 8.799x local speedup. The latter includes 829 cycles decoding weights
and activations, 424 GEMV cycles and 33 finalization cycles. These boundaries include
timestamp overhead; there is no assumed clock frequency. Standalone extraction
of 2,048 bytes takes 3,070–3,071 vector cycles versus 30,734 scalar cycles.
This is not a complete projection, inter-PE reduction, overlap or model token rate.

Both jobs succeeded and released normally without cleanup errors: compile
`wsjob-hwtzlyzcypfouevqvpgjxb` (41.416 seconds stage wall) and runtime
`wsjob-jgkalcjj5cwrnfhdbym5u2` (71.417 seconds stage wall). Maximum compiled SRAM
plus the unchanged 4,096-byte stack allowance is 18,208 bytes. A fresh account and
system audit confirms no owned active job or assignment. Simulator failure 001
(reserved parameter name) and complete simulator 002 remain frozen.

The separate compact weight-bank planner binds all 1,251 original tensors,
including 498 matrices and 400 FP8 scale aliases. Under the explicit rows=2 policy,
853,616 PEs hold compressed matrix banks and 16,384 PEs are reserved for other
actors. The bank estimate includes replicated FP32 scales, four descriptor bytes
per tile, 6,144 code bytes, 4,096 stack bytes and 1,536 scratch bytes. Its maximum is
47,536 bytes under the 48,128-byte ceiling, leaving only 592 bytes. **This is an
estimated ownership/storage result, not compiled whole-model SRAM admission.**
Actual routes, descriptor encoding, code size, non-matrix state and complete
execution remain to be implemented. Ten source tests pass, including independent
cyclic-ownership enumeration and cross-matrix local-offset uniqueness.

## Next evidence

Compile a resident bank actor with actual buffer/code extents; expose tile
predecode and operand receive as separate readiness events. Then measure regional
GEMV, numerically qualified reduction and overlap before complete-model integration.
No component milestone satisfies the full-model 2,000 tokens/s target.

## P4: compiled resident-bank capacity and predecode lifecycle (simulator)

`fp8-bank-sim-001` compiles the maximum candidate occupancy: 112 FP8 tiles,
12 BF16-sized reserved tiles,124 descriptor-sized words,112 FP32 scales, two
65-word operand buffers, one decoded 2x128 tile, decoder scratch and the executable
FP8 dot path. Both 2-PE program variants pass SRAM admission; the maximum actual
low-section end plus 4,096 stack bytes is **46,768 bytes**, leaving 1,360 bytes.
This is an actual component ELF result, not complete bank-runtime or model SRAM
admission. BF16/metadata capacity buffers hold sentinels; BF16 execution, full
scheduler and route code are not included.

All112 dynamically selected slots, including alternating boundaries, execute
against the previously frozen original-weight fixture. Inline and predecoded
outputs are bit-identical and within the independent bound. The complete weight,
scale, BF16-reserve, metadata and both packet buffers retain their expected values;
call counts and normal shutdown pass. The decoded buffer requires a ready token
for its matching slot, consumed by compute before reuse.

Same-PE simulator measurements separate the work explicitly: inline compute
982 cycles, or prefetch497 plus compute537 cycles. Including the inline variant's
10-cycle no-op prefetch, their measured sums are992 and1,034 cycles. **No overlap
or net speedup is established.** The operand's exact half encoding is prepared by
the test driver; a real on-wafer producer is not qualified by this probe.

The bounded workstation service completed in60.687 seconds and its lock is free.
No new hardware allocation was used for P4. Its immutable source and complete
simulator evidence are preserved separately from physical P3.

## P5: physical regional dataflow and measured overlap

The checked regional plan now lowers to column input multicast, ascending-K row
sums, row-result return and an all-row completion join. Explicit receive/decode
readiness gates native FP8 arithmetic. Independent packet and sum DSR leases allow
transfer and computation to progress concurrently; sends release storage only
through their completion callbacks. The matched serialized variant performs the
same arithmetic and transfers. This is a restricted component compiler, not a
complete physical model scheduler.

`regional-gemv-hw-001` uses64 physical PEs in an8x8 rectangle. Four original layer0
projection slices cover gate/up/down matrices, boundary row/K blocks and fixed
zero, mixed, small-scale and large-magnitude activation patterns. Every K-prefix
matches an independent ordered FP32 emulator bit for bit; the final row return,
all encoded operand packets, all endpoint event counters and all retained inputs
pass. Both schedules produce identical results within the frozen independent
FP64 bounds (maximum error1.4662742614746094e-5). All8 epochs and normal stop pass.

For each of the four paired cases, **serialized2,599 cycles versus overlapped
2,188 cycles** at the original root PE means411 fewer cycles,15.814% lower region
completion latency (1.18784x rate for this component). The measured boundary
includes input encoding, weight decode, multicast, local arithmetic, ordered
reduction and all-row completion. It includes host-launch entry skew. The schedule
also changes callback placement, so savings are not attributed solely to fabric
transfer. Local decode takes501–513 cycles serialized and521–535 overlapped;
contention is measured rather than assumed free. No cross-PE time subtraction or
nominal clock conversion is used. Host launch through result/audit transfer is
separately retained (about2.81–3.35ms); this is not a model-token rate.

All64 compiled PEs pass SRAM admission, maximum13,232 bytes including4,096 stack.
This contains one FP8 tile per PE. It is not combined full-bank/model admission.
Both jobs succeeded and released normally with no cleanup errors:
`wsjob-uudmpdnnxnpsvj8i55zcjm` compile42.416s stage wall, and
`wsjob-bmjrupukcwit69jtbyjydh` runtime199.952s stage wall. Runtime infrastructure
initialization accounts for much of that interval; it is not device compute time
or a claim about provider billing. Fresh job/system accounting confirms no owned
active job or assignment. The workstation simulator also ended normally, with
configured2GiB/no-swap/64-task limits observed while the service was live.

The prerequisite simulator passed in181.474 seconds. Source tests now total13,
including independent multicast, row-return, sum-neighbor and ack-neighbor endpoint
traversals and collision rejection. No frozen failure or functional baseline was
changed, and no tolerance was loosened after device observations.

Next: qualify fast dynamic activation encoding/scaling, combine resident banks
with actual role-specific communication footprints, and lower complete contraction
widths with local hierarchical reductions. Then integrate the complete original
model and qualify dependent sentence throughput. The2,000 tokens/s target remains
unmet; this partial projection milestone does not redefine final acceptance.

The supplementary `fp8-encoder-identity-001` checks53,725 finite FP32 bit patterns,
all finite FP8 centers and midpoint neighbors against an independent nearest-level
oracle. The direct-bit CSL encoder is explicitly uncompiled and unqualified on
device. This host mathematical check alone establishes no inference speedup.

## P6: physical direct encoding and unchanged dynamic quantization

`fp8-encoder-hw-001` passes all53,725 finite IEEE input patterns against both an
independent nearest-level oracle and PyTorch FP8 conversion. Coverage includes all
finite FP8 centers, midpoint ties and neighboring FP32 values, exponent extremes,
signed zeros and50,000 seeded random draws (duplicates/nonfinite draws excluded).
It also passes44 complete group128 quantizations. The maximum, scale multiplication
and input/scale division convention are unchanged; only the scalar binary search
encoder becomes direct IEEE rounding. Every output byte matches the original CSL
implementation and PyTorch, and every tested scale is bit-identical (0 ULP).
Inputs and untouched output tails are retained, and normal stop succeeds.

For the complete direct-encoding workload, measured original/candidate cycles sum
to7,713,018 /2,164,536 (3.56336x). This mixture includes clipped, tiny and normal
values and is not a model activation distribution. Complete group timing includes
maximum, scale, division and encoding. Across44 groups, the median changes from
46,315.5 to22,790 cycles; ranges are31,644–46,624 versus10,524–22,811. The ratio of
summed group cycles is2.04512x. No clock frequency or model token rate is inferred.
The remaining roughly22,800-cycle serial group cost motivates distributing
quantization across PEs rather than assuming the direct encoder meets the final
per-projection latency budget.

The first simulator attempt001 hit its240-second run deadline after successful
partial batches; no accepted completion is claimed for it. Attempt002 retains all
3,912 boundary/exponent critical patterns and a fixed4,096-input subset of the
random-draw set (the smallest bit patterns), totaling8,008, plus all44 groups. It
completes in152.410 seconds. Its exact same driver runs all53,725 direct patterns
on physical hardware. The numerical criteria and complete physical coverage were
not reduced to accommodate simulator throughput. Both snapshots remain frozen.

Physical compile `wsjob-pk88wvexelreoahsyr7rrp` and run
`wsjob-b78vfftydotk2wgwx8udpn` succeeded and released normally without cleanup
errors. Stage wall times were41.370 and202.087 seconds, including initialization;
they are not compute latency or provider billing. Maximum compiled SRAM including
the unchanged4,096-byte stack allowance is13,248 bytes. A fresh account/system
audit confirms no owned active job or assignment. The reusable encoder's header
now points to P6; the executed source is retained verbatim in its evidence snapshot.

## P7: composed resident banks and communication (simulator)

`regional-bank-sim-002` combines compressed resident banks, dynamic slot addressing
and the P5 communication/arithmetic protocol in one executable. The2x3 region
covers root, producer, row-root and ordinary compute roles. Their FP8 capacities
are108,110 and112 tiles, each with12 BF16-sized capacity buffers and corresponding
32-bit descriptor-sized slots. Role-specific capacity is explicit in the plan;
this is not yet a full-model ownership/routing assignment.

The first composed compile001 failed the unchanged SRAM gate: root48,176 bytes
and ordinary compute48,272, including the4,096-byte stack allowance. It was stopped
before simulation and never used hardware. Attempt002 removes unused per-stage
profiling and retains same-root whole-region timing. No weight capacity, stack
reserve or SRAM ceiling was reduced/raised to make this repair pass.

All six actual compiled PEs now fit:47,696–48,080 bytes including stack, only48
bytes of minimum remaining margin. This cannot be treated as space for uncompiled
BF16 execution or a complete scheduler. Those paths require their own actual
admission and potentially a different bank/role assignment.

First, middle and last bank selections across four original projection fixtures
pass in both schedules,24 epochs total. All K-prefix outputs match the independent
ordered FP32 emulator exactly and the FP64 bounds. The complete compressed weight,
scale, BF16-reserve and descriptor buffers are read back and verified, alongside
packet/slot/raw input checks and all protocol counters. BF16/descriptor buffers
are retained capacity sentinels; their computation/dispatch is not implemented.
Normal shutdown and workstation lock release are verified. Total bounded run time
is232.622 seconds; no new hardware allocation is used for this milestone.

Simulator root timings retain the first serialized9,882-cycle observation, with
other serialized observations1,752 and overlapped1,567. Launch-entry skew remains
in scope; no outlier is removed or full-model rate extrapolated. P7 is chiefly a
combined SRAM/lifecycle qualification, not a physical performance claim.

## P8: physical spatial quantization and direct native consumer

`spatial-quant-hw-001` runs the full group128 maximum, original scale multiplication,
original FP32 division and qualified FP8 encoding across32 PEs in an8x4 rectangle.
Four values belong to each PE. A maximum tree, one scale multicast and an ordered
payload tree produce the complete65-word operand packet at the root. The root
immediately invokes one original layer0 gate2x128 native FP8 tile, with no host
operation between producer and consumer. This is one partial dot, not a complete
projection, resident bank selection or full model.

All44 frozen P6 PyTorch groups pass. Every local output byte and scale bit is
exact, as are all32 subtree packets including unused tails. Every native product
matches the independent ordered FP32 emulator exactly (zero signs canonicalized)
and the frozen FP64 bound; maximum FP64 error3.799237310886383e-5. Input and original
weight/scale retention, endpoint counters, repeated epochs and normal stop pass.
The prerequisite simulator passes its16 predetermined groups in79.969 seconds;
no failed attempts or relaxed acceptance criteria are hidden.

Physical same-root producer-ready cycles span2,118–2,492, median2,484. They include
local maximum, maximum reduction, scale broadcast, encoding, ordered gathering
and the scale-send completion credit. The native consumer is519 cycles, making
complete producer-plus-consumer latency2,637–3,011, median3,003. Packet availability
is measured separately (median2,422). No nominal clock or tokens/s is inferred.
For context P6's single-PE producer median was22,790 cycles on these same groups;
that earlier interval excludes packing/gathering and is not a matched paired
schedule benchmark. P8 uses32 PEs rather than one.

Before each group, the driver launches receive arming and verifies every endpoint
is ready. This host barrier is excluded from device timing and separately reported
(median1.277ms including audit). Host start through result/audit transfer is also
separate (median2.717ms). One constant original weight tile is decoded before the
groups (500 device cycles). The device producer does not include an upstream
projection or input placement. These exclusions must be removed or integrated
when qualifying a self-driven complete model; they cannot be silently subtracted
from model end-to-end throughput.

Maximum compiled SRAM including4,096-byte declared stack is14,272 bytes. This is
an actor containing one weight tile, not the full resident-bank footprint. Compile
`wsjob-ehgytpojumajnkusihenmf` (81.526s stage wall) and runtime
`wsjob-erexmvfbee5z5pykbhzhad` (71.595s stage wall) both succeeded and released without
cleanup errors. A fresh account/system audit confirms no owned active job or
assignment; those wall intervals are not provider billing or compute latency.
The bounded workstation service is inactive, its main process exited and shared
lock is free. Fifteen source tests now cover independent tree traversal, ordered
partitions, physical links and resource collision rejection as well as prior work.

Next work remains full-width hierarchical reductions, actual BF16/native paths,
role-specific full weight ownership and device-driven epoch orchestration before
complete64-layer/full-vocabulary integration. The2,000 dependent tokens/s target
is still unmet; this milestone does not reduce its scope.
