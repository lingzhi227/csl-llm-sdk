> Current target is the resident spatial implementation of one continuing
> multi-turn conversation, at>=2000 average output tokens/s under
> [DIALOGUE-ACCEPTANCE.md](DIALOGUE-ACCEPTANCE.md). Earlier milestone targets and
> pending snapshots below retain their historical scope; aggregate independent-
> request throughput cannot qualify the current target.

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

## P9: complete K-width hierarchical contraction

The checked contraction plan covers all40/48/136 original K blocks of selected
layer0 gate, linear-attention output and MLP down-projection rows. These correspond
to5120/6144/17408 input columns. There are two real output rows per matrix; full
input width does not mean every matrix output or the full model is executed.

A static preorder binary tree has at most seven dependency edges at136 blocks.
At most two input queues receive child sums; fixed per-level/per-side colors give
non-overlapping express routes. Intermediate routers forward without software
relay tasks. The224 active tiles occupy disjoint serpentine subregions within an
8x28 rectangle. Tree and descending-K chain controls share exactly the same local
native dot work and have separate frozen FP32 aggregation references and FP64
absolute-product bounds. See FULL-K-CONTRACTION.md for resource/timing contracts.

Simulator001 compiles and admits the initial136x3 padded layout, but hits its
240-second run limit before a complete numeric record. Its exact progress location
was not logged and is unproven. Attempt002 removes184 padded PEs through the2D
mapping and adds phase logging, without changing logical trees, full K coverage,
arithmetic or acceptance bounds. It completes both schedules and all input/weight
retention checks in209.422 seconds total. No unchanged failed experiment is rerun.

In simulator002's frozen mixed case, chain/tree root cycles are4,893/1,898 for40
blocks,5,580/1,927 for48, and13,748/2,202 for136. Every native dot and every subtree
or chain-suffix result exactly equals its own ordered FP32 reference, all counts
and packet contents agree, and all FP64 bounds pass. Maximum observed FP64 error
is1.3412403276902296e-7. Actual maximum ELF SRAM including4,096 stack is12,432 bytes.
The workstation service exited normally and its shared heavy-job lock is free.
These observations are simulator component evidence; physical qualification is
recorded separately when its complete checks and allocation release are available.

### Physical P9 qualification

`contraction-hw-002` passes all four frozen cases in both schedules, eight complete
epochs in one runtime. Every local dot, every tree subtree and every chain suffix
matches its respective ordered FP32 reference exactly; the independent FP64 bounds,
participant counts, packets, callbacks and full weight/scale/input retention pass.
Normal stop completes. Maximum FP64 error across both schedules/all nodes is
0.00017780065536499023 in the large signed-value fixture; no tolerance is fitted
to observed output.

| Complete K blocks | Chain root cycles | Tree root cycles | Component speedup |
|---|---:|---:|---:|
|40 (5120 columns)|5100–5102|1965–1972|2.586–2.595x|
|48 (6144 columns)|5815|1994–2000|2.908–2.916x|
|136 (17408 columns)|14389|2274–2275|6.325–6.328x|

These same-root intervals include local operand encoding, weight decode, native
dots, full-K reduction and launch-entry skew. Separate host arming/readiness is
excluded from device timing. All sender callbacks are audited before rearming,
but the root interval ends at numerical readiness. Already-quantized activation
fixtures and only two output rows per matrix remain explicit scope limits. This
is not P8-plus-P9 fusion, full-M execution, a full model or a token rate. Regrouping
changes summation order; both paths satisfy their own frozen numerical contract.

Physical001 compiled successfully (`wsjob-td9p7dj3agghajcn3z9s8e`,192.527s stage
wall) and admitted all224 PEs, maximum12,432 bytes including the4,096-byte stack.
Its runtime `wsjob-cf4hvhxx2qncukelzr7ykb` failed before neural execution because
the service could not find the compiled artifact; the supervisor cancelled and
released it (53.767s stage wall). The1,131,713-byte archive exceeded the old1MiB
component upload chunk. Physical002 reuses the exact hash-pinned artifact/CSL
and fixture with an explicit8MiB single-message profile, without recompiling.
A no-job recording stub verifies byte-identical protobuf framing to the original
installed SDK; the prior profile used two messages. This successful repair does
not independently prove the server's internal failure mechanism on001.

The component reuse supervisor checks prior successful/released compile evidence,
all CSL hashes, artifact SHA, every ELF hash and SRAM again before the new run.
Runtime002 `wsjob-bodpssysjsslbcadkfddkr` succeeds and releases normally in71.493s
stage wall, without cleanup errors. A fresh account/system audit finds no owned
active job or assignment. Stage wall intervals are not device latency or final
provider billing. Original functional runtime/code and all failed attempts remain
unchanged. Seventeen source tests and the no-job actual-artifact framing proof
pass; the full original model and2,000 dependent tokens/s remain unachieved.

## P10: physical mixed resident FP8/BF16 banks

`mixed-bank-hw-001` closes the reserved-but-unexecuted BF16 gap. Every PE holds112
original FP8 tiles and12 original BF16 tiles while executing either native path
and P9's tree-only reduction. BF16 expands only two weights at a time to FP32,
using DSR7 independently from pending child receives. Checked4-byte type/slot
control replaces unused per-tile metadata; complete-model address ownership is
still unimplemented. The actual maximum footprint is47,472 bytes including the
unchanged4,096-byte stack allowance, leaving656 bytes below48,128.

The sixPE/2x3 physical component passes12 epochs, independent exact FP32 local and
subtree results, frozen FP64 bounds, complete callbacks, exact replay and all bank
bytes after one initial upload. The744 original tiles include BF16 head, untied
embedding and recurrent input weights alongside FP8 gate/out/down weights. First,
middle,last slots, zeros, tiny normal values, large patterns and alternating types
are covered. BF16 subnormal arithmetic and embedding lookup are not qualified by
this test. Maximum FP64 absolute error0.00390625 belongs to the largeBF16 case and
passes both the frozen bound and exact orderedFP32 gate.

Root component timing is1,252 cycles for FP8,1,213 for BF16 and1,217–1,245 for mixed
PE types. Timing includes resident dispatch, local arithmetic and the sixPE tree;
initialization, host operand/control transfer and receive arming are separate.
Different operands prevent a precision-speedup comparison. This is not fullK,
fullM or model token throughput.

Compile `wsjob-cgptn3ebb4wdhvr3grwbdp` and runtime
`wsjob-jfpmomx3qw4gztsme62zh4` both succeed and release normally. Their stage wall
times are41.451 and81.573 seconds; provider-billed hours are unavailable. Fresh
account/system accounting shows no owned allocation. Simulator002 passes in111.014
seconds and releases its bounded service/lock. Simulator001's builtin-name alias
compiler failure is preserved;002 changes that alias only and reuses the exact
frozen fixture. Nineteen source tests pass. See MIXED-BANK.md for implementation
and acceptance boundaries. Full-model2,000 dependent tokens/s remains unmet.

## P11: physical arrival-triggered resident epochs

`epoch-bank-hw-001` removes the per-epoch host barrier from the representative
mixed bank. Six full112-FP8/12-BF16 workers, one controller and one passivePE form
a2x4 component.66-word multicast requests trigger local computation; four-word
reductions carry exact contributor counts and monotonic epoch tags. Send completion
grants the next receive, and each root result gates the controller's next request.
Explicit microthread assignments are checked separately from queue/DSR identities.

Two96-epoch loops with rotated reset order pass on physical WSE-3. All4,608 local/
subtree FP32 values match frozen independent oracles, every subtree oracle passes
its pre-dispatch FP64 bound, and both complete root histories satisfy direct FP64
checks. Counts, tags, terminal callbacks, replay and all weights/oracles/requests
retain their expected values after one initial upload. Maximum actual SRAM is
47,792 bytes including4,096 stack and192 validation-oracle bytes per worker; the
48,128-byte ceiling and112/12 bank capacities are unchanged.

The full96-epoch loops take169,455 and169,474 same-controller cycles, averaging
1,765.156 and1,765.354 per epoch including real controller gaps. Request-to-result
medians are1,694 FP8 and1,659 BF16; gaps are89–90 cycles. Initial readiness and
terminal host auditing are separate. These are independent preloaded neural
operands with completion-dependent control, not autoregressive tokens or complete
projections. Full original model2,000 tokens/s remains unmet.

Compile `wsjob-xdtlrbvuw59bp9svown5yh` and runtime
`wsjob-rq9kf3ruklkqnnaalifpga` succeed and release normally, with41.533 and198.691
seconds stage wall including cluster setup. Fresh resource accounting reports no
owned active job or assignment. Simulator002 completes in140.875 seconds. Failed
simulator001 exposed concurrent send/receive defaulting toUT2 at the controller;
002 changes resource assignments and retains the identical numerical fixture.
All sources, diagnostics and release receipts are preserved. Twenty-one source
tests pass. See RESIDENT-EPOCHS.md for protocol and measurement limits.

## P12: complete original-model ownership and lifetime atlas

All1,251 original tensors,498 matrices,64 layers,96-position state and1,332 values
have explicit physical owners or storage intervals. The750x1160 fabric contains
860,880 FP8 bankPE and9,120 actorPE.36,864 banks cohost32x32 FP32 GDN shards;
824,000 non-state banks form the BF16 ring.620 compact whole-K dispatch segments
cover all matrices. Maximum bank data reservation35,256 bytes still requires
actual composed code/stack/scratch admission. Distributed values, normalization
gains, canonical weights and optional full diagnostic capture are explicit.

The independent auditor imports no builder code and visits all105,052,160 real
tile addresses. Complete geometry, original tensor identity, all state axes,
canonical byte slices and4,851 overlapping value lifetime pairs pass. Workstation
runtime4.757seconds, peakRSS422,316KiB under2GiB/no-swap/twoCPU/240seconds and shared
lock. Service and lock release are verified. No hardware is allocated; a fresh
account/system audit reports no owned job or assignment.25 source tests pass.

This is an ownership and addressing result. New state arithmetic, routing, actual
SRAM and complete token feedback are not admitted. Full-model2,000tokens/s remains
unmet. See COMPLETE-MODEL-ATLAS.md and frozen atlas-audit-001 evidence.

## P13: physical complete recurrent state co-resident with FP8 banks

`gdn-bank-hw-001` qualifies the atlas's4x4 state partition on20 physicalPE:
16 state workers each hold32x32 FP32 state and111 original FP8 tiles; one controller
and three passivePE complete the rectangle. Arrival-triggered prediction, delta
broadcast, state update and output reduction preserve state across8-step batches.
All96 positions plus8 reset-replay positions pass13,312 exact FP32 output checks
and13 full terminal-state checks against the frozen block oracle and unchanged
FP64 bounds.192 native FP8 dot values and full original bank retention also pass.
Inputs are preprocessed synthetic packets; convolution/gates/norm and full-model
feedback are outside this component.

Vector joins/delta and exclusive scratch reuse reduce the component's stable
simulated step from14,445 to6,628cycles in preserved partial runs. The accepted
simulator008 completes all gates in228.232seconds with packed32-bit transport of
unchanged weights. Physical steps measure6,755–6,758cycles, median6,756.5, with
58-cycle gaps. This is not a physical paired speedup or model token rate.

Actual maxSRAM including4,096stack is46,784bytes, margin1,344.19 compiler images
cover20 application coordinates exactly; a new component gate checks PT_LOAD
rectangles and retains the original SRAM ceiling. Compile and runtime both succeed
and release normally (41.374/81.434seconds stage wall). Workstation service/lock
and physical account/system release are verified. All failed/partial attempts
remain frozen;27 source tests pass. See COHOST-RECURRENCE.md. Full-model2,000tokens/s
and complete route/state/nonlinear integration remain unfinished.

## P14: complete projection lowering and physical route-word transitions

All 498 original matrices lower into 306 shared-input bundles and 980 events.
FP8 input preparations decrease from 400 to 256; all tensor identities and rows
remain present. Transitive reduction retains 18 additional memory completion
edges, independently protecting all 192,736 overlapping-storage value pairs.
Four complete abstract reduction forests cover FP8 K40/48/136 and BF16 K40.
The independent full audit checks 44,766,384 PE/color entries, compact profile
masks, all physical holes and exact matrix dispatch. `projection-audit-003`
passes in 2.873 seconds at 197,256 KiB peak RSS without reading weights or
allocating hardware. BF16's longest single tree edge is 797 physical hops, an
explicit unoptimized layout cost. See PROJECTION-LOWERING.md.

`route-epoch-hw-001` separately qualifies the SDK-derived compact route backend,
per-color teardown/completion latches and input/output queue rebinding. A packet
feeds back across a reversing 20-PE path for 64+8 autonomous epochs. Every payload,
tag and counter gate passes, as does full 705,120-byte synthetic sentinel-bank
readback. Same-endpoint send/return measures 1,132–1,141 cycles, median 1,139;
this includes two traversals and a transition, not isolated switching or model
throughput. Max compiled SRAM plus declared stack is 48,112 bytes, only 16 bytes
below the unchanged ceiling. It does not include neural arithmetic code.

One earlier simulator compile failed because direct task-29 binding conflicts
with SDK2.10.1's shared memcpy teardown dispatcher. The accepted simulator uses
per-color handlers, with unchanged packet/storage gates. Sources, failure and
release receipts are retained. Both physical jobs succeeded and released; the
fresh job/system audit finds no own allocation. All workstation resources are
released and 32 source tests pass. See ROUTE-WORD-QUALIFICATION.md.

General forest transition readiness, complete operand multicast/output scatter,
combined neural-kernel SRAM and full-model 2,000 dependent tokens/s remain unmet.

## P15: paired column input and physical original mixed-bank consumers

A new complete overlay retains all498 original matrices and105,052,160 real
tiles. K40/K48 use the existing860,880-bank ring; K136 uses857,208 banks in a
748-column subring. The BF16 ring is unchanged except its aligned phase. Every
state/KV/value/canonical owner stays fixed. Maximum bank payload remains35,256B.
Independent audit002 checks all addresses and8,700,000 dense input PE/color
entries in9.307seconds at528,744KiB RSS. The preserved001 dead-end failure leads
to terminating horizontal routes at their last recipient. Payload counts for
K40/48/136 are260/520/780words at the busiest direction, not measured latency.

A20-PE two-stage counter-window component passes physically: horizontal streams
select130-word pairs, vertical bank filters select65-word operands, and arrivals
trigger unchanged original FP8/BF16 native dots. Twelve full mixed banks exercise
110FP8/13BF16 and111/12 profiles. Ten cases cover both horizontal periods, every
pair offset, both vertical offsets, zero/tiny/boundary operands, first/middle/last
slots and reset/replay. All240 native values,11,700 delivered packet words,
callback/teardown counts and421,560 original payload bytes pass exact gates and
independent FP64 bounds. The unified transfer also verifies1,512 padding bytes.

Physical SRAM with4,096-byte declared stack is47,328bytes, margin800. Local
receive-to-dot completion is1,026cycles FP8 and992BF16; upstream delivery and
reduction are excluded and no model rate is claimed. Compile/runtime both
succeed and release; an independent system audit confirms zero own allocations.
The accepted simulator completes in132.991seconds; one earlier readback timeout
and two pointer compile errors remain preserved. All36 source tests pass.
See COLUMNAR-INPUT.md and filtered-bank-summary-001.json.

Full-matrix routes and output return, source quant/value delivery, bidirectional
vertical execution, new-class reduction forests and combined runtime SRAM remain
the next gates. P14 owner descriptors cannot be reused unchanged with this
overlay. Complete original model2,000 dependent tokens/s is still unmet.

## P16: complete class binding, physical composed bank and exact shift decoder

All original projection dispatch descriptors now explicitly name the P15
storage classes. Four regenerated forests independently pass 44,718,648 PE/color
entries, all 105,052,160 original tiles, graph/arena dependencies and FP8 input
consumer/color checks. Final audit002 completes in 3.613 seconds at 241,900 KiB
RSS. Plan002 corrects inherited base census/aggregate metadata without changing
any graph, dispatch or dense route output; earlier plan/audit001 remain frozen.
No full-matrix route execution is implied by these metadata checks.

`coupled-bank-hw-001` composes two-stage operand filtering, original native
FP8/BF16 arithmetic in twelve maximum mixed banks, ordered subtree reduction
and return to the source controller. All ten cases pass 240 local values,
240 subtree values, 20 returned values, full packet/callback/teardown checks,
421,560 original payload bytes and 1,512 padding bytes. Same-controller complete
component intervals are 1,685–2,499 cycles (median 1,911), excluding host arming
and final audit callbacks. This is partial-K independent-operand execution,
not a complete matrix or model throughput measurement.

The no-temporary FP8 decoder reduces nine vector operations to five. Exhaustive
host and physical comparison checks all 65,536 packed patterns with no bit
mismatch, plus the original native-dot and tree numerical gates. Matched physical
decoder intervals are 250,563 versus 130,498 total cycles, a 1.92005x local gain.
Both original and new decoder sources remain available. Actual compiled SRAM
including the declared 4,096-byte stack is 48,032 bytes, leaving 96 bytes.

Two early simulator attempts expose stale reads when child DMA buffers alias
decoded scratch; separate receive buffers pass without relaxing the numerical
gate. The exact cause is not established. All five attempts and diagnostic
receipts remain frozen. Final simulation passes in 205.996 seconds. Both
physical jobs succeed and release normally, and fresh account/system and
workstation-lock audits show no owned allocation. All 39 source tests pass.
See COUPLED-BANK.md and coupled-bank-summary-001.json.

Full matrices, real quant/value delivery, BF16 input distribution, many-root
output return, generic epochs, all-layer feedback and complete-model correctness
and 2,000 dependent tokens/s measurement remain the integration objective.

## P17: physical native shape/scaling sweep and complete original BF16 matrix

`native-shapes-hw-002` qualifies four equal256-MAC shapes (2x128,4x64,8x32,16x16)
and a matched vector-output-scaling copy of each shape on eight physical PEs.
Three vector multiplies preserve the original FP32 scaling order and rounding
boundaries. Eight frozen inputs and twelve timing modes pass720 active outputs,
zero tails,6,240 packet words,3,072 counted native invocations, matched-pair
identity and282,048 original bank bytes, followed by normal shutdown and release.
Actual maximum SRAM including4KiB stack is46,560bytes, leaving1,568bytes before
future communication integration. All45 source tests pass.

For the same16x16 original FP8 tile, scalar/vector mean intervals are683.03125
and327.15625cycles:2.08778x locally. Including each call's weight decode gives
935.25 versus580.28125cycles:1.61172x. The equal-MAC shape sweep favors8x32 for
FP8 and16x16 forBF16 locally, but different shapes cover different original
matrix slices and have different reduction/communication costs. No complete
matrix or model speedup is inferred. See NATIVE-SHAPES.md and the frozen summary.

The complete original layer0 in_proj_a48x5120 candidate now lowers all960 tiles
at their original atlas addresses, with24 K40 trees and an ordered48-value return.
Independent route/owner audit003 passes all7586 PE/color entries. Simulator017
passes all native/subtree/full outputs,62,400 input words,122,880 retained target
words, deliberate one-bit negative controls/restoration and normal stop. Its
maximum actual SRAM plus stack is48,048bytes; its interval is9,328simulatorcycles.
All earlier compiler, memory, timeout and crash attempts remain preserved.

Physical compiler001 succeeds/releases but its host wrapper rejects a measured
12,170,645-byte archive against the old8MiB cap. Attempt002 reuses the exact
archive after source,1262PE coverage,SRAM and16MiB one-message protobuf admission.
Its first complete original matrix case passes all outputs/inputs/counters at
10,824physicalcycles. The next arm/fence stalls; timeout cancels/releases the job
before full retention, replay and normal stop. Two zero-bank diagnostic simulator
epochs subsequently pass, without establishing the physical fault's cause.
Attempt003's one-channel host transport then passes all six cases, exact replay,
11,520 native and11,520 subtree values,288 returned values,374,400 input words,
33,832,240 original bank bytes plus13,520 padding bytes, and normal stop. All
five full original BF16 matrix calls measure10,824physicalcycles; the remaining
FP8 cohost smoke case uses different original weights and measures10,633cycles.
Compiled SRAM plus stack remains48,048bytes. Both jobs succeed and release.
The exact earlier stall cause remains unproven; its unchanged first-case device
interval shows that one-channel host transport is a reliability change, not
arithmetic acceleration. See FULL-MATRIX.md and full-matrix-summary-001.json.
Complete-model2,000dependenttokens/s remains unmet.


## P18: physical compact2D complete matrix and shared projection lowering

The first8x32 retile keeps the old631x2 strip and passes all simulator/retention
gates, but worsens its interval from9328 to10242cycles. A retained-fixture audit
proves identical full BF16 matrix/inputs and unchanged background slots. This
negative result is preserved; the independent strip hardware trial is deferred.

The mainline now has parameterized physical K/M axes, explicit input/output
owners, stream/completion/buffer/resource documents, an independent route/traffic
audit and a shared matrix-epoch CSL protocol.40x24 workers plus40 edge input
owners/controller form41x25 PEs. Static maximum input distance is24hops and
maximum directed-link traffic66words including teardown. These counts are not
a complete credit-critical-path model or a performance prediction.

Simulator001 passes; physical001 exposes pair reordering on the third epoch
although native/subtree/counter checks pass. The queue retains its previous final
gather color. Resetting the first color during arm is qualified with two nonzero
simulator epochs and physical002's complete six-case/replay/full-bank run.
All11520 local and11520 subtree values,288 returned values,374400 input words,
producer packets, callback/teardown counts and33845760 bank/padding bytes pass.
All five full BF16 cases measure3898physicalcycles vs10824:2.77681x,63.99% lower
latency. The matched FP8 smoke measures3694cycles; it remains representative
cohost data, not a complete original FP8 projection. All jobs release normally.
Actual maximum SRAM plus4KiB stack is48112bytes;109 images cover1025PEs exactly.
All55 source tests pass. Failed attempts and source hashes remain preserved.

This is a restricted single-projection backend. Input operands are pre-positioned
at40 owners before the timed kick, and48 outputs go to a qualification sink.
The next boundary is a real complete-dimensional multi-region model subgraph,
not further isolated matrix tuning. The initial importer binds real layer0 MLP
nodes13..18, shared quantization, actual successor interfaces and cross-operator
lifetimes, but has no admitted physical placement/routes/executable schedule.
Complete original model, correct sentences and>=2000dependenttokens/s remain unmet.

## P19: complete-dimensional MLP ownership and spatial fusion candidate

The new136-region candidate maps every original MLP matrix tile and scale index
to paired gate/up workers, local activation actors and40 independent down output
stripes. The independent audit covers1,044,480 original tiles and702,880 known
stream paths. Actual code/SRAM, complete routing, distributed readiness and
full-subgraph execution remain unqualified. Treating all helpers as bank-free
would exceed the prior full-model actor allowance by5784, so geometry alone is
not storage admission.

Packed fusion helpers keep projection, SiLU, multiply and residual BF16 boundaries
and direct native input encoding. Simulator003 passes1350 exact words against
the existing reciprocal-multiply quantization convention. The earlier compiler
failure and literal-division scale mismatch remain preserved, with both oracles
reported. All workstation services release; no new ALCF jobs are submitted.
See MLP-SPATIAL-FUSION.md for scope, resource gaps and the next executable boundary.
No complete model or >=2000dependenttokens/s result is claimed.

## P20: full original residency candidate and physical resident chunk joins

All1251 original text tensors, all64 MLP layer slots, complete embedding/head,
GDN/KV/conv state and norm gains have compact addresses anchored to the new MLP
coordinates. The independent audit enumerates870000 profiles,36864 GDN shards
and103232 auxiliary pages, and proves exact matrix stream/domain coverage.
5800 embedding-only collector/header helpers eliminate the old bank-free actor
deficit at the metadata capacity level. Maximum allocated payload is35232bytes;
other combined role programs, dynamic intermediates and communication remain
unqualified.

The actual resident helper program forwards the original-format65-word ingress
packet, joins ordered FP32 partials in8-word credited chunks and performs original
embedding lookup. Simulator001 and physical002 pass three warm epochs,1152
intermediate/final values,512 lookup words,390 relay words, all callback counters
and complete original-bank/padding retention. Only two internal helper instances
with one color parity are qualified. Synthetic partial sources are not the actual
down projections, and host launch scaffolding is not a full-model epoch controller.
Maximum physical SRAM plus4KiB stack is46128bytes, leaving2000bytes.

Physical001 compiles but fails on host import before any runtime job. Its failure
is preserved; physical002 fixes only driver entry/backend selection and reuses
the exact artifact/CSL/ELFs/fixture. The one compile and one runtime job both
succeed and release. Final account audit has151 terminal own jobs, none active
or assigned. All69 source tests pass. See MLP-RESIDENCY-CHUNKS.md and the bound
summary receipt. This progress does not satisfy complete MLP timing or full-model
correct dependent sentence generation at>=2000tokens/s.

## P21: connected static MLP source and selected role compiler admission

Separate static down routes remove the candidate native route rewrite and its
row-fence/readiness exchange. An independent full-dimensional fabric audit checks
1059920 flows and4843946 PE/color definitions, with256 maximum aggregate words
per directed link for the listed MLP and two RMS phases. This is not a cycle
prediction or proof of runtime liveness.

The real worker, fused activation/quantization, resident collector/header, norm
and spectator-bank bodies are implemented. The input boundary performs a real
residual add; output chunks feed the real following RMS. Their changed distributed
sum orders and complete numerical execution remain unqualified. Selected25-role
compile004 passes47776bytes including4KiB stack, leaving352bytes. All71 source
tests pass. The full870000-PE candidate layout is7470bytes of source.

Full compile001's host syntax error and002's600s compiler timeout are preserved.
Their resources are released. Optimized whole-wafer compile003 is live at the
milestone snapshot, with4GiB/no swap,2CPU and1800s/1850s bounded compile/service
limits. No complete layout, numerical runtime or timing result is inferred. No
new WSE jobs are submitted; the latest hardware audit finds no own active or
assigned jobs. See MLP-STATIC-PIPELINE.md and its bound summary. The complete
model, correct dependent sentences and>=2000tokens/s target remain unmet.


## P22: resident spatial stages, fused interfaces and request ownership

All1251 original tensors and1172 operations now lower to66 disjoint stages:
embedding,64 layers and full head. Eight78-wide macro columns contain alternating
north/south layer order while the overall model advances west to east. Internal
mix/gate-up/down rectangles are adjacent, capacity-derived2D partitions. Exact
native tile/page ownership, conservative values and two isolated request states
fit the planning budget; composed CSL SRAM remains unqualified. New original
BF16 row granularity avoids unused bank tails without changing weights.

The generator emits192 layer-local fusion contracts,65 adjacent stage interfaces
and9944 one-hop port pairs across all declared internal/inter-stage interfaces.
These counts do not include complete region-internal distribution or establish
all-route/color legality. Numeric fusion retains required full-K, RMS, group128
and BF16 boundaries. Python protocol tests exercise backpressure, opposite
completion orders, partial/duplicate/stale traffic, warm resets, real token
feedback dependency and finite context. The CSL lease guard is source-only.

The superseded full-wafer shared-layer compile003 was verified by PID, working
directory and unit, then owner-cancelled. Logs, sources, cancellation and empty-
cgroup receipt are retained. No new physical job or speed result is claimed.
See ARCHITECTURE.md and the pipeline-stage-map-002 evidence. Next work is actual
connected layer0/1 execution in this layout, followed by full64-layer feedback.

## P23: resident native loops and selected fused backend compiler admission

The unchanged disjoint stage rectangles now have executable native-loop addresses
for all498 original matrices and115077120 tiles. Each PE retains fixed K inputs
and streams one partial output block. Paired8x32 gate/up and4x64 down shapes reduce
conditional native busy costs under measured primitive reuse; new-loop/full-stage
latencies remain unmeasured. Original BF16 rows remain1x128, and GDN states use
4x8 FP32 pages with independent request ownership.

The direct fused body accepts actual complete-K packed BF16 gate/up pairs, then
performs SiLU, multiply, original group128 quantization and credited native down
fragments. Low-payload resident PEs host the larger fused actor. Stage request
controllers occupy proved unused local PEs, fixing attempt004's SRAM overage.
Selected six-role compile008 passes at47712bytes including4KiB stack, leaving416.
Earlier syntax/resource failures and earlier accepted005/006 compositions remain
frozen. This is not all-role/full-layer SRAM admission or fabric execution.

Original layer0/1 packing audit001 passes6080 sampled tiles across all16 matrices,
complete12 small parameters and selected original publisher shard hashes. The
new weight stream never changes original precision. All97 source tests pass,
including the11 evidence-scoped prediction tests. The supplied P22 predictor
still lacks full-stage costs and clock calibration, so it supplies no full-model
TPS result and does not silently treat P23 native-loop costs as measured latency.

All eight bounded compiler services and the weight audit are released, no WSE
jobs were added, and the final account snapshot has no own active job/system
assignment. See RESIDENT-LAYER-BACKEND.md and resident-layer-backend-summary-001.
Next work remains actual connected layer0 -> layer1 routing and neural execution;
complete64-layer correct sentences and>=2000 aggregate generated tokens/s are
still unachieved.

## P24: arrival-driven full-K producers and fused block receipts

Real native projection workers now receive/retain original K inputs, consume
ordered full-K subtree packets, round the complete projection and release each
row only after local send plus actual consumer credit. The full64-layer MLP
reduction audit covers2752 contractions,502320 workers,499568 tree edges and
2122720 PE/color entries. All139264 projected blocks map to the original fused
group owners. Ingress, root-to-consumer and external credit routes remain open;
these reduction paths alone do not connect the layer graph.

The512-byte decode arena is shared across disjoint input/compute/reduction/send
phases. A single ordered child DMA frees DSR5/UT4 while retaining local-left-right
addition. Selected16-role compile007 passes47936bytes including4KiB stack. Exact
original matrix plus live auxiliary payload is preserved in every selected
profile; dummy unused tail bytes are not allocated. Earlier over-budget compiles,
unsupported builtin and pointer-cast failure are frozen. The changed direct-single
native path and complete reduction order remain numerically unqualified.

The fused actor consumes11-word actual projection packets and immediately issues
a copied-block receipt, allowing the producer's next rows to arrive before the
whole128-value quantization group completes. Its receipt buffer survives until
send completion. This interface compiles in backend009 but has not yet run as a
connected physical stream. All102 source tests pass. Every new bounded compiler
service is released; no WSE job was added and the final hardware snapshot has no
own active job/system assignment. See LAYER-PROJECTION-NETWORK.md and the bound
layer-projection-network-summary-001 receipt. Complete layers and full-model
correct sentences at>=2000 aggregate generated tokens/s remain unfinished.

## P25: whole-region projection-to-fusion transport

Both original gate/up geometries compile with every actual native/reduction and
fusion role:5226/5264 PEs,1957/2847 ELF images, maximum47712bytes including4096
stack. Fixed-color routes003 checks4032 projected streams and139264 original
blocks across64 resident layers. Private DMA source ownership and hardware FIFO
backpressure remove per-block software receipts; a separate early suffix buffer
prevents cross-root head-of-line waiting. Original bank addresses and arithmetic
orders are unchanged. Unsupported WSE-3 color swapping, failed fixed-color
receipt routing and syntax failures remain frozen. All109 source tests pass.

This is regional compiler/protocol admission. Input/downstream distribution,
stage drain, changed-operation numerical qualification, neighboring-layer and
full-model sentence generation/TPS remain unfinished. No WSE job was submitted;
all eight P25 bounded compiler services are released and account allocation is
empty. See LAYER-FUSED-TRANSPORT.md and layer-fused-transport-summary-001.json.

## P26: connected original MLP compiler admission

The original5120 ->17408 ->5120 MLP now has actual on-device connections from
spatial input quantization through native gate/up, fused BF16 activation/group128
quantization, down distribution/reduction and full output collection. Forty
input quantizers,33 fused actors and10 segment sinks cohost low-payload native
PEs without moving original banks. Tagged input routing preserves full and
ragged-tail K ownership. Both enclosing regions compile with mixer/state banks
resident but inactive:11388/10998 PEs,4179/4894 ELF images,maximum47776bytes
including4096 stack,352bytes minimum margin. All115 source/protocol tests pass.

Real WSE-3 compilation rejected simultaneous RAMP/cardinal RX. The corrected
exclusive-grant bus selects one input and restores transit only after output
queue flush. Earlier syntax failures and the180s bounded timeout remain frozen;
005/006 finish within the revised300s compile cap. All six workstation services
are released,631864899bytes retained remotely,zero WSE jobs added and no owned
hardware allocation at the final snapshot.

This is connected-component source/compiler admission, not neural execution or
maximum-speed evidence. The granted bus needs measurement and likely sharding.
Changed arithmetic/whole-MLP numerical qualification, actual complete local and
request drain, residual/norm/mixer/attention, neighboring layers and original
full-model sentences at>=2000 aggregate tokens/s remain unfinished. See
LAYER-MLP-NETWORK.md and layer-mlp-network-summary-001.json.

## P27: physical complete original MLP fusion

Physical003 passes the complete original5120 ->17408 ->5120 MLP on11,388 PEs:
four warm cases,20,480 bit-exact BF16 outputs, every native input/scale and
completion counter, all resident bank/table retention and complete local drain.
The independent row-major oracle separately bounds each projection against FP64
truth; PyTorch supplies SiLU and validates actual FP8 quantization. An exhaustive
CPU proof covers all65,280 finite BF16 inputs using a6,656-byte resident SiLU table
per fusion actor. Physical results cover the actual full-MLP input cases.

Nonzero controller counts are2,844,472/2,844,301/2,844,443; zero is2,358,523.
SDK loading323.486s, original initialization3.122s and full diagnostic runtime
340.388s are reported separately. The actual SRAM maximum48,128bytes includes
4,096 reserved stack and has zero margin. No model throughput is inferred.

Compiler001 succeeded/released but host ELF-file admission failed; runtime002
failed client serialization before neural execution and was cancelled/released.
003 reuses the same artifact with bounded bank reads and passes normal shutdown.
No CSL or numerical criterion was changed between those physical attempts.
All122 source tests pass, all owned resources are released, and the original
functional baseline remains preserved. Full layers/state, neighboring stages and
complete original-model sentences at>=2000 aggregate generated tokens/s remain
unfinished. See LAYER-MLP-QUALIFICATION.md and its bound summary receipt.

## P28: parallel input preparation and vector communication

Physical004 retains all P27 original-weight numerical, ingress, counter,
retention, warm-reuse and normal-drain checks:20,480 exact BF16 outputs. Forty
initial quantizers prepare independently before exclusive transmission, inside
the controller timing window. Controller native-slice and output copies use
vector operations. Fused activation quantization was already arrival-triggered;
this corrects the prior proposed optimization without changing the P27 result.

Nonzero controller counts fall to713,625–713,752, about74.91% fewer than P27 and
a3.985–3.986 counter ratio. The explicit completed-output host interval is
2.388–2.801ms; no calibrated wall speedup or model TPS is claimed. New phase
markers and a frozen static controller-network traffic audit guide the next
coalesced-send and overlap work. All127 tests and all11,388 PE SRAM gates pass,
with unchanged maximum48,128bytes including4,096 stack. Both fresh jobs succeed
and release; no workstation heavy job or model payload retransfer is needed.
The full64-layer correct-sentence and>=2000 aggregate tokens/s goal remains
unmet. See LAYER-MLP-PERFORMANCE.md and layer-mlp-performance-summary-001.json.

## P29: coalesced packets and one-frame prefetch

Physical005 keeps all strict four-case original-MLP numerical, ingress,
retention, drain and normal-stop checks. Spatial lowering exposes complete
native distribution packets and independently verifies full/tail selectors.
864 small sends become176 group packets with unchanged49,376 wire words;
independent receive/packet/grant leases permit one prefetched response.

Nonzero controller counts are613,127–613,190,14.07–14.10% lower than P28.
Completed-output host observations are2.673–2.969ms and all slower than P28;
no end-to-end wall speedup or model TPS is claimed. All176 next grants issue
during packet leases, but zero next frames arrive before the lease retires.
This limits the observed prefetch benefit and motivates native/distribution work.
All132 source tests and11,388 PE SRAM gates pass, with unchanged maximum48,128
bytes including4,096 stack. Both new jobs succeed/release; no workstation job
or payload retransfer is added. See MLP-COALESCED-PREFETCH.md and the immutable
mlp-prefetch-milestone-001.json. Complete layers/model-speed acceptance remains open.

## P30: shared full/tail native input

Physical006 preserves full original-MLP numerical/ingress/retention/drain/normal
stop qualification while one selector-coded wire copy serves full and ragged
native owners. Hardware range filters and local part normalization halve native
broadcast words49376->24688, with unchanged343040 consumed operands and176 group
packets. Nonzero controller counts fall to564289–564333,7.965–7.972% below P29.
Completed-output host observations2.551–2.834ms are shorter than P29 but remain
longer than P28; no sustained wall speedup or complete-model rate is claimed.

All134 source/publication tests and11388 physical PE gates pass; maximum48128
bytes including4096stack leaves zero margin. Both physical jobs and all five
workstation services release. Paired-input native simulation regresses on used
shapes, and a locally faster unroll exceeds full-layout SRAM on68PE; both are
preserved without hardware trials. See SHARED-NATIVE-INPUTS.md and the bound
shared-input-milestone-001.json. Residual/norm/device handoff, adjacent complete
layers, complete original sentences and>=2000 aggregate tokens/s remain open.

## P31: complete residual/RMS and MLP graph admission

The new arrival-driven graph includes both full5120 residual/RMS operations,
forty adjacent norm/quantizer pairs and the original full MLP, with no host
intermediate neural uploads. Compile010 rejects40 combined actors at51872B;
compile011 rejects80 split actors at49264B. Compile012 relocates800 explicit
128B recurrent-state pages two rows north while conserving all396953216 resident
bytes and every native matrix coordinate. All11388 PE gates pass, maximum48112B
including4096 stack. Reference002 and139 source tests pass. All four workstation
services release. Physical007 is dispatched; physical numeric, timing and final
resource receipts remain pending. See NORM-MLP-BRIDGE.md. This is graph/compiler
admission, not full-layer, full-model or model-speed acceptance.

## P39: complete original physical mixer projections

P39 passes176 source and176 publication tests. Physical `layer-mixer-hw-002`
executes all five complete original QKV/Z/A/B/output projections through3554
internal endpoints. All86400 stored BF16 outputs are bit-exact to the predeclared
independent reference, with zero/change/replay, all counters and20 warm drains.
Every396953216 original bank byte and every descriptor is retained; normal stop
and release are confirmed. Physical compilation covers11388 PEs/7714 images,
maximum48128bytes including4096 stack. Download admission is now scoped to the
measured102028613-byte complete-mixer artifact. Four cluster jobs (one actual
WSE runtime) and both workstation services are released; failures are preserved.
Host root read/consume and full retention timings are diagnostic, not model TPS.
Actual conv/GDN consumers, complete neural stages and the multi-turn target remain
unfinished. See [COMPLETE-MIXER-QUALIFICATION.md](COMPLETE-MIXER-QUALIFICATION.md).

## P40: one-conversation storage and turn admission

P40 specializes the complete layer00 banks for one continuing conversation,
retaining all original weights, the complete first context and both work slots.
Fresh compile021 admits
all11388 PEs/7910 ELF images, maximum48112bytes including4096 stack. Actual SRAM
falls by3207168bytes; remote reference001 independently verifies every retained
word in393746048bytes. A turn-admission oracle checks exact history, pending final
tokens, three continuing turns, reset/overflow and all-stage retirement. Twenty
selected source and20 publication tests pass; these are protocol/storage checks,
not device dialogue or speed results. Two failed/cancelled compiler attempts are preserved. All four
workstation services release and no hardware job is submitted. See
[Single-dialogue banks](SINGLE-DIALOGUE-BANKS.md).

## P41: fused root packets and frontend compiler admission

P41 implements direct fused projection packet output and16 shared-Q/K
convolution/gate frontend groups. All33 selected cohost programs compile; all16
consumer groups keep their complete original bank extents, with maximum47984bytes
including4096 stack. Routing checks cover all16480 projected values and1408 paths.
The new resource planner rejects the unchanged whole-stage placement:1275
estimated matrix-prefix conflicts require joint weight/communication placement,
not only state-page movement. New numerical execution, the recurrent core and
complete layer/model inference remain unfinished; no speed claim is made. All
six workstation services release and no new hardware job is submitted. See
[Frontend fusion](FRONTEND-FUSION.md).

## P42: compact original banks and complete frontend admission

P42 admits the complete original layer00 banks with native packet fusion and
all16 shared-Q/K frontends: full compile022 covers11388PE/7919ELF, maximum48112
bytes including4096stack. Remote reference001 verifies all454400 mixer tiles,
28605 retained pages and every original source word;433136 identical scale
aliases save1732544bank bytes without requantization. All existing MLP/norm/control
routes remain. Joint row/network placement reduces merge actors1216->168;
this is not a measured speedup.34 selected source and34 publication tests pass. All six workstation
services release and no new WSE job is submitted. Numerical frontend execution,
recurrent connections and complete multi-turn model speed remain unfinished.
See [Compact frontend](COMPACT-FRONTEND.md).

## P43: physical original frontend histories and native-arrival arithmetic

P43 physically qualifies all16 original frontend groups/48 heads over six
continuous positions and two reset-replay positions:148224 FP32 packet values,
49152 gated BF16 outputs,245760 exact history samples,259512 native packets and
all43104 retained parameter words pass. Shared Q/K and reset replay are exact.
This standalone32-PE test injects independent original projection values and
frozen recurrent results; actual GDN, a complete layer and model speed remain
unqualified.40 selected source and40 publication tests pass. Both cluster jobs
and all seven workstation attempts release normally. See
[Frontend numerical qualification](FRONTEND-NUMERICS.md).


## P44: exact original MLP scale sharing and complete-bank admission

Complete compile024 admits11,388 PEs/8,428 ELF images with maximum48,032B
including4,096B stack. Exact local scale sharing removes3,912,128 bank bytes;
the actual complete census saves3,362,352 SRAM bytes. Independent remote
reference002 verifies all98,003,376 original source words,1,044,480 MLP tiles
and28,605 retained pages from actual descriptors. Existing routes, original
weight codes, arithmetic order and both work slots remain unchanged.

Paired calibration032 covers20 PEs. Compile023's600-second timeout and recorded
2/3/4GiB memory limits remain preserved; bounded024 completes in520.129s.
Twenty selected source and twenty publication tests pass.
All services release; no new WSE allocation. GDN math-only029 and rejected
SDK-replacement030/031 are separately scoped. This is a storage/compiler
milestone, not new numerical execution or model speed. See
[Compact MLP](COMPACT-MLP.md) and [Recurrent integration](GDN-INTEGRATION.md).

## P45: complete recurrent co-placement and original-word preservation

P45 fits the actual recurrent-port programs and all original layer00 banks on
11,388 PEs/8,240 ELF images: maximum48,112B including4,096B stack, with16B minimum
margin. Joint MLP output placement reserves753 state workers holding all786,432
FP32 recurrent elements. Original-bank reference003 verifies all97,025,344 source
words,1,044,480 MLP tiles and7,085 nonrecurrent pages; banks occupy388,094,976B.
Compile026's five112B overflows are resolved by moving ten auxiliary pages.
Forty-five selected source regressions, seven targeted refinement checks and
46 final publication regressions pass. All seven workstation services release;
no new WSE job is submitted. Actual GDN broadcast/returns, changed numerical
execution, full-layer inference and multi-turn model speed remain unqualified.
See [Recurrent co-placement](GDN-CO-PLACEMENT.md).

The complete027 compile takes849.522s under the corrected16GiB/1,800s policy.
All48 heads retain complete128x128 FP32 state;748 cohosts own eight columns each,
and five own32 each. Every old state coordinate has an explicit inverse mapping.
Final placement moves908 nonrecurrent pages between PEs versus P44, including
the ten-page refinement, while both work slots remain. The bank proof flushes
and rereads all source values through actual compiled native descriptors.

The scalar return currently differs from P43's paired frame contract and is not
routed. Prefer a separately qualified paired variant for these even-width
workers. Complete MLP output rebalancing moves16 gate/up rows and12 down rows
between full and ragged K trees; changed neural execution requires fresh bounds
and reference comparisons. Preserve failed033, rejected034, unselected035,
timeout025 and overflow026. This milestone claims compilation and storage only.

## P46: physical full-state GDN with native paired returns

P46 physically qualifies all 48 recurrent heads on their 753 original value
slices using paired native returns. Six continuous positions and two reset-replay
positions pass: 6,291,456 FP32 state elements, 49,152 BF16 outputs and 24,576
five-word return frames. Fresh capture checks prove bitwise reset replay; maximum
state relative L2 error is 3.202e-7. Physical compilation covers 1,506 diagnostic
PEs/758 ELF images, maximum 26,992B including 4,096B stack. Twenty-seven selected
source and 27 publication regressions pass, with four targeted builder checks.
Both cluster jobs and all three workstation services are released. This uses
frozen original-reference frontend inputs and diagnostic state banks; actual
frontend connections, the complete resident neural layer and dialogue speed
remain unqualified. See [Paired recurrent qualification](PAIRED-GDN.md).

The exact worker/math sources match the selected simulator smoke and complete
diagnostic compiler artifact. The independent FP64 oracle fixes propagated
FP32 bounds before candidate output; observed state error reaches at most
0.164656 of its per-element bound. The maximum predeclared bound norm ratio
is 0.000065025, below the 0.002 admission cap. These are component criteria,
not permission for token divergence in complete-model acceptance.

Preserve the SDK-color compiler failure001 and intentional simulator stop002.
The corrected smoke003 passes all 6,144 selected state elements and 24 frames.
Physical001 then covers all original slices. Runtime initialization takes
236.494s; the complete diagnostic driver takes 240.282s including host transfers,
full state readback and verification. These timings are not neural latency or
model TPS. Full production routes, launch/return fences and all64 layers remain
open. Original functional code and the P45 complete-bank snapshot remain intact.
