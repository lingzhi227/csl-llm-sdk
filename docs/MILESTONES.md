# Accepted milestones

### Original full-model parameters prepared · First 20-layer stage compiled · September 23, 2026 UTC

All original decoder layers 0–63, final normalization and the complete
248,320-word output head now have independently checked BF16 preparations.
Original rows are packed directly into bounded transfer arrays and decoded
against independently reread original bits, including every padded halfword.
The original parameters and their dimensions remain unchanged.

The complete original layers 0–19 compile into an actual WSE-3 artifact:
24,743 programs cover 690,200 PEs and 8,957,066 coordinate banks. Complete
program-index verification independently checks coverage, layer identities,
bank addresses and static SRAM. The minimum margin is only 48 bytes against
49,152 bytes including the declared 4,096-byte stack. All 149 failures of the
earlier family-specific ceilings remain disclosed. Dynamic stack peaks, task
DSR lifetimes, full-stage host-copy binding and neural execution are unqualified.
Full 64-layer physical inference and dependent token generation remain open.
[Report](SEQUENTIAL-PREPARATION-STAGE0-COMPILE.md) ·
[Source](../examples/sequential_stages) ·
[Evidence](../evidence/sequential-stages/qualification.json).

### Original four-layer chain · Fresh full-state restore accepted · September 23, 2026 UTC

Original BF16 layers 0→1→2→3 execute causal hidden handoffs entirely on one
physical WSE-3, at positions 0 and 1 plus device reset and position-0 replay.
All 48,084,480 conditional matrix rows and 375,660 predetermined exact FMA
samples pass, together with operator, state, transport and parameter checks.
A distinct physical runtime reloads original weights and the position-0
checkpoint, then computes position 1. All 39 semantic banks, 14,378,752 native
bytes, match uninterrupted position 1 bit-for-bit.

All 36,789 raw files from both successful runs are durably preserved and hashed.
Nominal CPU parity is not claimed: the maximum absolute hidden difference across
the four layers is 0.0078125. Conditional operator checks are not a propagated
whole-chain error bound. Failed compilation/runtime/audit attempts remain in
the evidence history. The restored run's largest parent-monitor poll gap is
60.486867 seconds, so a hard 15-second cancellation guarantee is not established.
Full 64-layer/full-vocabulary generation and matched performance remain open.
[Report](FOUR-LAYER-FRESH-RESTORE.md) · [Source](../examples/four_layer_chain) ·
[Evidence](../evidence/four-layer-chain/qualification.json).

### Original vertical Layer3 · Physical attention and reset accepted · September 23, 2026

The complete original-weight Layer3 executes positions 0/1/reset/0 in a
29×1160 region: 30,576 matrix PEs, 24 attention heads and 136 MLP owners.
All 11,741,184 conditional matrix rows, 91,728 predetermined exact FMA samples,
state, transport and head-archive checks pass. The three final hidden vectors
are bit-exact with the earlier accepted horizontal Layer3 outputs. Original
nominal BF16 differences remain 346/1,150/346 of 5,120 values, with maximum
absolute errors 0.00390625/0.001953125/0.00390625. Reset replay is bit-exact.

All 11,341 raw files are durably preserved. The compiler succeeded; an original
host inspector incorrectly applied Layer0's SRAM policy and failed. Its source,
failure and report remain preserved. Corrected inspection of the same artifact
passes the already-admitted Layer3 policy, without recompilation. This is finite
context-8 conditional qualification, not full-model generation or performance.
[Report](LAYER3-VERTICAL.md) · [Source](../examples/layer3_vertical) ·
[Evidence](../evidence/layer3-vertical/qualification.json).

### Original vertical Layer0 · Physical positions and reset accepted · September 23, 2026

The complete original-weight Layer0 executes positions 0 and 1, then device
reset and position-0 replay, in the accepted 30×1160 top-spine region. All
12,114,432 conditional matrix-row checks, 94,644 predetermined exact FMA
samples, full 128-wide recurrent replay and complete state/transport/reset
checks pass. Original nominal BF16 differences are 0/438/0 of 5,120, with
maximum absolute differences 0/0.0009765625/0. The complete 608-array neural
reset replay is bit-exact. These are conditional operator and state results,
not a propagated whole-layer bound, full-model generation or token benchmark.

The third attempt uses 16 channels and an 8 GiB remote worker. The original
240-second startup timeout and subsequent HTTP502 failure at 4 GiB remain
preserved; their unique cause and a minimum worker-memory requirement are not
established. Physical and offline owners exit normally and device release is
independently verified. All 5,312 raw files are durably preserved and hashed.
[Report](LAYER0-VERTICAL-PHYSICAL.md) ·
[Host/audit source](../examples/layer0_vertical/physical_runtime) ·
[Evidence](../evidence/layer0-vertical/physical-qualification.json).

### September23,2026: complete original Layer0 vertical static qualification

The top-spine30×1160 program passes actual compiled resource checks for34800 PEs
and31548 original matrix tiles. Packet-tree hops fall from the superseded bottom
layout6287604 to2819067 for the same messages; this is a source geometry metric,
not measured latency. Actual SRAM plus4096-byte reserve peaks at47824/48128.
The16-channel peripheral is independently matched byte-for-byte to the prior
vertical compile. [Source and scope](LAYER0-VERTICAL-TOP.md) preserve the
bottom regression and its correction. Layer0 physical numerical execution,
checkpoint control restore and full-model inference remain open.

Only accepted results are listed below. Unresolved integration work and the remaining scope are tracked in [implementation status](STATUS.md).

## Dense transport with arithmetic · Finite physical qualification accepted · September 23, 2026 (UTC)

The synthetic 136-PE fixture passes positions 0/1/reset/0, three stable full
snapshot pairs, retained parameters, all complete rows and buffer ownership.
Independent reconstruction of actual operands checks 1,179,648 ordered FMAs,
9,216 collective additions and 3,072 BF16 outputs with zero mismatches. Runtime
and audits exit normally; resource release and immutable backups are independently
verified. The held-ACK release RPC meets its host deadline, with later source ACK
checked separately. Earlier compile/startup/log-export/diagnostic failures remain
preserved. No original-weight layer, model or token-performance claim is made.
[Report](DENSE-TRANSPORT-FINITE136.md) ·
[Source](../examples/dense_transport/finite136) ·
[Evidence](../evidence/dense-transport/finite136/summary.json).

## Layer0 arithmetic · Finite nine-PE simulator qualification accepted · September 22, 2026 (UTC)

All 35,181 finite exp inputs and six synthetic convolution/DeltaNet/gated-RMS
computations pass the original budgets, full 128 ordered FP32 checks, persistent
state, zero resets and exact replay. Independent reconstruction checks 294,912
FMAs and 98,304 state words without numeric or sign mismatches. This uses shared
qualified integer primitives, not a second arithmetic engine. Conditional FP64
differences remain reported; no continuum, complete-layer or CS-3 claim is made.
The original SDK export failure and 240-second SIM001 timeout remain preserved.
[Report](LAYER0-ARITHMETIC-FINITE9.md) ·
[Source](../examples/layer0_arithmetic/finite9) ·
[Evidence](../evidence/layer0-arithmetic/finite9/summary.json).

## Original Layer3 · Continuous positions and device reset accepted · September 22, 2026 (UTC)

One initialization and weight upload serve physical positions 0 and 1, followed
by device reset and an exact position-0 replay. All three captures pass the
conditional operator gates, KV/state checks and independent reconstruction.
The device client exits normally. The original outer supervisor exits 1 because
its schedule comparison mixes a JSON list and Python tuple; that failure and
absence of COMPLETE remain preserved. A separate postcapture audit passes.
Nominal CPU BF16 mismatches are 346/1,150/346 of 5,120; no propagated whole-layer
enclosure or bitwise parity is claimed. Full-model inference remains open.
[Report](LAYER3-CONTINUOUS-RESET.md) ·
[Source](../examples/native_layer3/continuous_reset) ·
[Evidence](../evidence/native-layer3/continuous-reset/summary.json).

## Original Layer3 · Single-position physical/operator contract accepted · September 22, 2026 (UTC)

The complete original Layer3 graph runs position 0 on physical WSE-3, with
normal exit and independently verified raw data, transport and parameter
retention. All conditional operator gates pass; independent exact FMA samples
use a different row at every matrix PE. The final BF16 vector differs from the
nominal CPU reference at 346 of 5,120 values (maximum absolute error 0.00390625).
There is no propagated whole-layer error enclosure or bitwise parity claim.
Repeated positions, reset, all 64 layers and text generation remain open.
[Report](LAYER3-STATIC-TRANSPORT.md) ·
[Source](../examples/native_layer3/static_transport) ·
[Evidence](../evidence/native-layer3/static-transport/summary.json).

## Full136 static transport · Physical qualification accepted · September 22, 2026 (UTC)

The 534 PE fixture passes all 136 READY messages, 8 requests, 24 fragments and
1,024 halfwords. Two complete exports match; source/relay/sink ownership and
local causality checks pass, followed by normal exit and independently verified
release and backup. This is one transport epoch without reset. Original neural
epochs remain zero, and previous native-control failures remain preserved.
[Report](STATIC-TRANSPORT-FULL136.md) ·
[Source](../examples/ready_fanin/static_transport/full136) ·
[Evidence](../evidence/ready-fanin/static-transport-full136.json).

## Small static transport · SDK qualification accepted · September 22, 2026 (UTC)

The 27 PE fixture delivers all 3 READY packets,8 requests,24 fragments and 1024
halfwords. Two complete exports match, all retained-input/provenance/release
checks pass and the runtime exits normally. The first capture-timeout failure
is preserved. Full 136-producer physical transport and original neural inference remain
unqualified; this is not token-latency evidence.
[Report](STATIC-TRANSPORT-SMALL.md) ·
[Source](../examples/ready_fanin/static_transport) ·
[Evidence](../evidence/ready-fanin/static-transport-small.json).

## Full136 actual sender-frame checks · Failure evidence accepted · September22,2026 (UTC)

All136 producers traverse three qualified actual frame/source/tail checks,
but RUN011 receives the same malformed prefix and still fails strict checking.
Retained fault context and peer records match RUN010; earlier origin ordering
differs. Point checks do not establish continuous DMA or wire immutability,
a unique routing cause, a transport repair or a neural epoch.
[Report](FULL136-SENDER-FRAME.md) ·
[Source](../examples/ready_fanin/native_control/sender_frame) ·
[Evidence](../evidence/ready-fanin/native-control/full136-sender-frame.json).

## Full136 origin word receiver · Failure evidence accepted · September 22, 2026 UTC

Physical RUN010 receives the same malformed prefix through qualified scalar
input-queue task arguments. Its packet order and retained context differ from
RUN009. Bulk origin body DMA is not necessary for this occurrence; strict
protocol checking still fails. All408 producer source records, normal exit,
release and backup are independently verified. No repair or neural epoch.
[Report](FULL136-WORD-RECEIVER.md) ·
[Source](../examples/ready_fanin/native_control/word_receiver) ·
[Evidence](../evidence/ready-fanin/native-control/full136-word-receiver.json).

## Full136 body-completion boundary · Failure evidence accepted · September 22, 2026 UTC

Physical RUN009 detects the same bad eight-word prefix as RUN008 before arming
the current packet's tail receive. The first 22 origin records, all 16 peer
records and all 408 producer source records remain exact and comparable.
Strict protocol checking still fails; normal exit, release and durable backup
are verified. Earlier boundary handling, routing, body DMA and sender lifetime
remain alternatives. No repair or neural epoch is claimed.
[Report](FULL136-BODY-BOUNDARY.md) ·
[Source](../examples/ready_fanin/native_control/body_boundary) ·
[Evidence](../evidence/ready-fanin/native-control/full136-body-boundary.json).

## Full136 first-fault capture · Failure evidence accepted · September 22, 2026 UTC

Physical RUN008 still fails the original 650-record criterion. All 408 producer
source records are exact and all 13 original RUN004 captures are byte-identical.
Two added fault-bank reads agree: origin's malformed eight-word prefix is retained at
an ordinary-tail failure in RX phase 3; the buffer suffix contains prior fragment
data. Normal exit, release and durable backup are independently verified. This
is diagnostic evidence, not a repair or neural-inference pass. [Report](FULL136-FIRST-FAULT.md) ·
[Source](../examples/ready_fanin/native_control/first_fault) ·
[Evidence](../evidence/ready-fanin/native-control/full136-first-fault.json).

## Bidirectional native control · Strict physical pass after prepared launch · September 19, 2026 UTC

Physical RUN007 passes all 31 checks over 9 packets/200 words, two assembled and
two released source rows, complete banks/suffixes, stable state and both first-TX
lease witnesses. Origin also records RX during unfinished TX. The only device
change moves the existing GO after REQUEST0 preparation. Normal exit, owner reap,
resource release and raw evidence are independently verified. One pass neither
identifies a unique root cause nor repairs the 136-producer failure;006 and the
simulator failures remain preserved. [Report](NATIVE-CONTROL-BIDIRECTIONAL.md) ·
[Source](../examples/ready_fanin/native_control/bidirectional/prepared_launch) ·
[Evidence](../evidence/ready-fanin/native-control/bidirectional-prepared-launch.json).

## Bidirectional payload and row lifecycle · Accepted subscope, strict run failed · September 19, 2026 UTC

Physical 006 completes 9 packets totaling 200 words, two synthetic 128-halfword rows, complete
banks/suffixes and stable 64-word state at all 3 PEs. Origin records one actual RX
during unfinished TX. Only 30/31 independent checks pass: producer first-TX witness
is 0x12a rather than 0x107, so the unchanged strict suite remains failed. Ten raw
captures, normal exit and independent resource release are verified. Full
SIM012/014/015 precompute failures and SIM013 prefix-only success remain preserved.
The 136-producer corruption and complete original neural epochs remain unresolved/zero.
See the [scoped report](NATIVE-CONTROL-BIDIRECTIONAL.md).

## Native control two-source comparison · Matched simulator and physical pass · September 19, 2026 UTC

Three PEs, two independent senders and one shared short route pass the same
four-packet73-word check on both backends. Full banks and retained suffixes,
per-source ordering, complete stable state and both first-packet overlap
witnesses are independently verified. Physical005 saves seven captures and
exits normally with actual owner reap and independent release. All actual
physical executable sections/task tables and DSR operations match019. The
018 parser failure and larger physical136 failure remain preserved. This is
transport qualification, not a neural epoch. See the [report](NATIVE-CONTROL-MULTISOURCE.md).

## Native control physical comparison - Accepted compile, failed runtime, September 19, 2026 UTC

All356 programs compiled and passed independent fit/allocation checks at27,872
bytes including stack. Physicalrun004 then failed strict checking with447
coherent records:408 exact producer records,22 valid origin deliveries and a
control-tail error on ordinary value3. Only initial/producer raw files match
the prior comparison; receive order and the first fault changed. Normal stop,
actual executor exit1 and independent release were verified. No repair or
original neural epoch is accepted. See the [report](NATIVE-CONTROL-PHYSICAL.md).

## Native control termination — Scoped SDK qualification, September 19, 2026 UTC

Separately recorded isolated positive/negative cases, consecutive 31→8 packets,
and the same sequence with actual SDK message routing pass strict data, lease,
order, suffix and finite stability checks. Routing-enable readback is exact on
both PEs. Original failed suites003/004/007 and source-rejected009 remain failed
or unexecuted; all actual owners were reaped and resources independently released.
This does not qualify the physical136-producer fixture or a complete neural layer.
See the [report](NATIVE-CONTROL-QUALIFICATION.md) and exact source provenance.

## Managed RX comparison — Same-system identical failure, September 19, 2026 UTC

The managed receive descriptor variant passed minimal syntax and complete
356-PE compilation with 25,600 bytes maximum including stack. All 13 actual
physical captures match combined-TX runtime002 byte for byte on the same system.
The strict protocol still fails at the same record and payload; all 408 producer
records are exact. Normal stop, release and actual waiter exit1 are verified.
No full-graph promotion, unique cause or complete neural epoch is accepted.
See the [report](READY-FANIN-REPRODUCTION.md) and immutable source provenance.

## Combined-frame TX comparison — Compile and failed-runtime evidence, September 19, 2026 UTC

The corrected combined-TX module passed two-PE syntax and complete 356-PE physical
compilation; maximum storage plus 4 KiB stack is 25,600 bytes. Physicalruntime002
still failed on the same malformed receive, with443 coherent records and all 408
source observations exact. It saved 13 captures and exited normally. Independent
release and actual waiter exit1 are verified. Different physical system assignment
limits causal comparison. No full-graph promotion, root-cause claim or original
neural epoch is accepted. Pointer/comptime type failures and the unexecuted003
draft remain in the [report](READY-FANIN-REPRODUCTION.md).

## Concurrent READY physical failure reproduction — Accepted evidence, September 19, 2026 UTC

The bounded 356-PE fixture reproduces a network-header value inside received
application data with 136 concurrent producers and no neural math. All 408
source point snapshots are exact. Strict checking fails at origin record 21;
13 raw captures and a normal context exit were independently verified. The SDK
predecessor timed out with status-only evidence and remains failed. Resource
release is confirmed for all attempts. Root cause, transport repair and complete
original neural epochs remain unaccepted. See [report](READY-FANIN-REPRODUCTION.md).

## Complete native graph and first-error evidence — Accepted scope, September 19, 2026 UTC

Both fullgraph001/002 pass actual placement, original storage and SRAM review:
1,254 programs cover all 33,750 PEs with no demotion; 47,552 bytes including the
4 KiB stack allowance leaves 576 bytes. Fullgraph002 adds a 44-byte first-error
record without changing mathematical kernels or packet transport. Two failed
physical prefixes independently verify 195 original parameter readbacks and all
24 head-finished metadata. Runtime003 captures READY_FORMAT with reserved word
0x00400108. The upstream cause, archive payload and full-layer numerics remain
unaccepted. Both attempts timed out during later diagnostic reads and were
released; no normal stop or complete neural epoch is claimed. See [report](NATIVE-LAYER3-GRAPH.md).

## Physical QK archive and selected-program fit — Accepted, September 19, 2026 UTC

The synthetic four-PE original-kernel fixture passed two resets, exact960-word
archive and256-BF16-output equivalence per reset, actual1280-word source overwrite,
61 rejected API calls and172 successful assertions. Thirteen durable captures
and43 journal events include normal stop; all owned resources were released.
A separate89-position complete-role compile accepted all24heads/24sinks, maximum
47552 bytes including4096stack, margin576. Its unselected programs are demoted;
it is not an executable full graph. The prior51424-byte fit failure and five
fixture failures remain documented. No original-weight neural-layer acceptance
or dependent decode is claimed. See [report](QK-ARCHIVE-PHYSICAL.md).

## Native KV/Q SDK transport — Accepted, September 19, 2026 UTC

The 45-PE synthetic fixture passed three normal operations including a reset,
29,184 exact payload halfwords, 192 canaries, deliberately held streams and actual
source/receiver ownership. It rejected 525 invalid lifecycle API calls. Thirteen
immutable captures, 47 journal events and normal stop were independently checked.
The runtime reused 63 accepted compilation files; peak memory was 220,430,336
bytes and guarded time 197.475793 seconds. All owned resources were released.
Three failed attempts remain documented. This is transport qualification, not
a complete original neural layer or performance result. See [report](NATIVE-KV-SDK.md).

## Original layer 3 FIFO diagnostics — Accepted diagnostic, September 19, 2026 UTC

A 16-PE SDK fixture qualified finite FIFO capacity/device-gated drain and packet
coexistence. Full original layer 3 compilation produced 563 programs/33750 PEs,
max 47920 bytes including 4096-byte stack. One physical 32 KB snapshot independently
retained 20 READY heads, 549of 576 Q/K/V packet observations and 380 FIFO events;
heads 17, 21, 22, 23 remain incomplete. Complete neural epochs: 0. All resources were
released. The aliased retained-RMS audit remains invalid. See[report](LAYER3-FIFO-TRACE.md).

## Full original MLP and full CPU reference — Accepted, September 18, 2026

HW01 completed the full 5120→17408→5120 original MLP on physical WSE-3 for four
inputs with independent numerical/retention/release checks. A separate full
64-layer CPU reference generated four tokens with original cache restoration.
Neither result is complete CSL model generation. See[MLP](HW01-FULL-MLP.md) and
[reference](FULL-REFERENCE.md).

## HW00 physical fragment — Accepted, September 18, 2026

Original layer3 resident 8PE MLP, three inputs, all125operations and112arrays passed independent numerical audit on physical WSE-3. New physical compilation/SRAM/placement, normal stop and device release were independently verified. Full MLP/model remain open. See [report](HW00-PHYSICAL.md).

## WP16 eight-PE resident fragment — Accepted after independent review, September11,2026 (UTC)

Three original dense/changed/zero inputs traverse resident192 ->128gate/up,
SiLU/product and128output MLP on4x2PEs. Independent numerical, placement,
handoff/state, complete initial/final weight and normal-stop checks passed.
SDK002's original guardexit1 is preserved: geometry-specific auxiliary identities
were qualified separately from saved artifacts. No rerun was needed. Scope is a
resident fragment, not full WP16 or physical three-system inference.
See [report](RESIDENT-SPATIAL-RUN002.md) and [evidence](../evidence/wp16-resident.json).

## WP16 connected partial MLP — Accepted, September 11, 2026 (UTC)

Four original-weight PEs execute three dense/changed/zero generations with resident
128×112 gate/up weights, device SiLU/product, 128×128 partial down and explicit
reset/retention/release barriers. Independent numerical, transport, generation
sensitivity and normal shutdown checks passed in221.818s under512MiB/Swap0/CPU0.
Full5120 connected reuse and whole17408-channel MLP remain open. This is a separate
profile from the full-width one-input result below.
See [report](WP16-CONNECTED-MLP-REPORT.md) and [evidence](../evidence/wp16-connected.json).

## WP16 partial MLP — Accepted, September 10, 2026

Four original-weight PEs connect full5120-column gate/up projections for128 rows,
wholeSiLU/BF16product and128×128 partialdown for one dense input. Independent
source/observedFP32cast/transport/retention/release/normalstop checks passed.
Full17408-channel MLP, connected reuse/reset and full-model inference remain open.
See [report](WP16-PARTIAL-MLP-REPORT.md) and [evidence](../evidence/wp16-partial.json).

## WP15 — Accepted, September 10, 2026

Ten PEs connect original selected projections, trained Q/K processing and persistent
KV attention for two original-hidden tokens in one runtime. See
[report](WP15-REPORT.md) and [evidence](../evidence/wp15.json) for exact coverage.

## WP14 — Accepted, September 10, 2026

One dense5120-column original projection input flows through four Q/K projection
PEs, exact device handoffs and trained RMS256/partialRoPE64 at position1 on a
fifth PE. Source, stage, cast, transport, post-consumer producer retention and
normalstop checks pass;512projection/512consumer official BF16 matches. This is a
selected Q/K call without V/gate attention or KV persistence.
See [report](WP14-REPORT.md) and [evidence](../evidence/wp14.json).

## WP13 — Accepted, September 10, 2026

Original selected layer3/head0 Q256/rawgate256/K256/V256 rows span all5120 input
columns. Eight separate single-PE four-call runs cover all1024rows with the same
four hidden inputs; independent source/rounding/state/identity/normalstop gates pass.
This is separate-runtime projection coverage, not a device attention consumer.
See [report](WP13-REPORT.md) and [evidence](../evidence/wp13.json).

## WP12 — Accepted, September 10, 2026

One PE composes original synthetic Q/K through RMS256/deviceRoPE64 into persistent
D256/cache8 attention and sigmoid gating, with device-only operand handoff. Ten
successes, overflow-before-preprocessing, reset, full source/stage/cache checks and
normal stop passed. All2560observed final BF16 values match the official fixture;
source intervals remain conservative, not a general bitwise guarantee. See
[report](WP12-REPORT.md) and [evidence](../evidence/wp12.json).

## WP11 — Accepted, September 10, 2026

One Q/K pair passed ordinary RMS256 and device partial RoPE64 for text positions0–7.
Ten calls match all5120official BF16 outputs;9600RNE boundaries,3840tail elements,
zero signs, fixed trig probes, reset and normal stop passed. Composition with
attention and long-context positions remain separate. See [report](WP11-REPORT.md)
and [evidence](../evidence/wp11.json).

## WP10 — Accepted, September 10, 2026

One-PE D256 attention with persistent BF16 cache8 and sigmoid output gating passed
ten tokens, capacity refusal, reset and normal shutdown. All2560outputs exactly
match the official BF16 reference. Inputs are already Q/K normalized and rotated;
all-head routing, projections and complete-layer integration remain separate.
See [report](WP10-REPORT.md) and [evidence](../evidence/wp10.json).

## WP08 — Accepted, September 10, 2026

Repaired one-PE selected-head preprocessing passed eight tokens, full width4
history, all numerical/conversion gates and reset checks. The first failed
beta-source/exp candidate is excluded; thresholds and fixtures were unchanged.
See [report](WP08-REPORT.md).

## WP07 — Accepted, September 10, 2026

Standalone BF16 direct-gain gated RMS128 and its third-PE recurrence composition
passed all declared stage/conversion, state and completion checks. Four calls
and four composed tokens fit the same SRAM/host limits. Only the observed
successful join order is device-qualified. See [report](WP07-REPORT.md).

## WP06 — Accepted, September 10, 2026

One full128×128 synthetic recurrent head passed four token updates across three
generations on two PEs. All prediction, delta, full-state and output gates passed;
state persistence, resets and framed transfer lifecycle were verified. Both PEs
fit48 KiB including4 KiB stack allowance. See [report](WP06-REPORT.md).

## WP05 — Accepted, September 10, 2026

Four full5120 ordinary RMS calls and64 signed conversion probes passed with
offset gain and actual BF16 RNE. Shared gain/output storage plus a4 KiB stack
allowance fits the48 KiB application budget. The first noncompliant SRAM
candidate was stopped and excluded. See [report](WP05-REPORT.md).

## WP04 — Accepted, September 10, 2026

Pinned source/metadata audit matched851 text tensors and established branch-specific
equations. Twenty-seven bounded CPU checks passed extracted official function bodies
against small independent references. Full runtime/model integration remains open.
See [report](WP04-REPORT.md) and [semantic contract](WP04-SEMANTICS.md).

## WP03 — Accepted, September 10, 2026

Original BF16 rows 0:128 span all 5120 input columns using device-owned persistent
FP32 accumulation across 46 tiles. Four calls passed all boundary/final, state,
nonzero-tail-padding and guard checks, including exact one-hot at global5119 and
zero-after-nonzero. Simulation completed in 205.8 seconds with normal stop/exit.
Scope is 128 outputs, not the entire 17408-row projection or model.
See [report](WP03-REPORT.md) and [evidence](../evidence/wp03.json).

## WP02 — Accepted, September 10, 2026

Two adjacent PEs split the original tile into two 56-column contractions, send
partial FP32 results over fabric, SUM on the receiving PE and normalize 128 elements
with unit gain. Four calls passed all independent stage/approximation/state gates
and exercised local-first and receive-complete-first joins. Normal SDK stop/exit
was recorded. This is an on-wafer reduced chain, not physical cluster inference.
See [report](WP02-REPORT.md) and [evidence](../evidence/wp02.json).

## WP01 — Accepted, September 10, 2026

An original 128×112 BF16 up-projection slice completed four calls in one runtime,
including changed input, exact last-column one-hot and zero-after-nonzero. All
weight/guard/counter/handle checks passed; normal SDK stop and process exit were
recorded. See [report](WP01-REPORT.md). This is a local GEMV tile, not full inference.

## WP00 — September 10, 2026

Resource admission, serial heavy-task locking and three host bit codecs; a single
WSE-3 PE copied seven BF16 raw bit patterns with native MEMCPY_16BIT on SDK 2.10.1.
H2D, device copy and D2H preserved `0000 8000 3f80 bf80 7f80 ff80 7fc1`.
SDK stop returned normally and both compiler/simulator processes exited zero.

Outer wall times were 4.083871 s for compilation and 2.058936 s for simulation.
These include container/service overhead and are not hardware performance.
The first attempt failed before compilation due to host/container mounting; the
controller approved a corrected launch with unchanged CSL/driver source.

This milestone establishes a tiny transfer control, not BF16 arithmetic, request
reset, packed-stream SDK support, model generation or physical cluster inference.
Fourteen lightweight host tests pass. See [WP00 report](WP00-REPORT.md) and
[compact evidence](../evidence/wp00.json). Future entries require scoped acceptance.
