# Complete original BF16 matrix integration

This candidate executes the original layer 0
`linear_attn.in_proj_a.weight`, with all 48 output rows and all 5,120 input
elements. Full-model inference and the 2,000 dependent tokens/s target remain
unqualified. Compilation and abstract routing alone are not numerical acceptance.

## Placement and dataflow

The pinned P12 atlas and P15 overlay place all 960 two-row by 128-column tiles
in BF16 slot 6, at eligible ranks 502,400 through 503,359. Original global worker
coordinates run from (420,705) to (120,706). The executable slice occupies
631 by 2 PEs, including 960 workers and a temporary adjacent source at
(119,705). The source's underlying full-model bank role is not admitted here.

Workers allocate 35,256 bytes each. Their exact candidate profiles are 110 FP8
plus 13 BF16 slots, except the wide-ring boundary with 84 FP8 plus 13 BF16
slots. The complete target matrix occupies its exact original slots. Other
slots contain representative original mixed-bank prefixes in physical fixtures;
they are not claimed to have their complete-model owner identities. The padded
full-bank transfer is 33,845,760 bytes.

One 2,600-word input stream crosses the physical serpentine path. Each worker's
hardware counter filter selects its 65-word packet. Twenty-four independent
ordered K40 trees reduce all local two-row results. A binomial concatenation
tree streams pairs from the 24 roots back to the source in original row order.
Root buffers retain only the last forwarded pair; callback ownership prevents
reuse before send completion. The source retains the full 48-value result.

The independent reverse-address and router traversal audit checks all 960
original owners, 936 contraction edges, 23 gather edges, 7,586 PE/color entries
and every directional neighbor. Input reaches exactly all 960 workers through
1,261 forwarding PEs, with a maximum path of 1,260 hops. The 44,520 summed
contraction/gather word-hops are a communication volume, not measured latency.

## Explicit resource ownership

| Worker operation | Queue | Microthread | Descriptor index |
| --- | --- | --- | --- |
| Selected input | IQ2 | 2 | 2 |
| Left native subtree | IQ3 | 3 | 3 |
| Right native subtree | IQ4 | 4 | 5 |
| Native or collected output | OQ2 | 5 | 6 |
| Gather child pair | IQ5 | 6 | 1 |
| Native arithmetic | Local memory | Synchronous | 4; BF16 also 7 |
| SDK host copies | SDK queues | SDK threads | 0 |

SDK 2.10.1's installed `memcpy/wse3/memcpy.csl` reserves destination and
source-1 descriptor index 0 by default. The initial gather candidate reused
destination 0, conflicting with live host progress copies. Attempt 010 was
cancelled after this static conflict was found; attempt 011 moves gather to
index 1. This is not established as the cause of the earlier initialization
timeouts. `spatial/sdk_leases.py` rejects explicit application use of default
SDK descriptor, queue and microthread reservations before subsequent staging.
It does not prove dynamic buffer lifetimes or queue liveness.

## Numerical and timing gates

The fixture reads all original matrix bytes from the pinned, publisher-hashed
layer 0 shard. An independent row-major FP64 matrix-vector product cross-checks
the per-tile mathematical oracle. Required execution gates cover every local
dot, every ordered K40 subtree, all 48 final values, every selected packet,
every root's retained pair and exact callbacks/teardowns. Ordered FP32 results
must match exactly after zero-sign canonicalization and satisfy independently
computed FP64 error bounds. Original target weights must be retained.

The simulator executes one mixed BF16 case and uploads only its complete
original target matrix; unused background allocation remains zero. Hardware
qualification additionally requires signed-zero, tiny-normal, boundary-normal,
replay and FP8 cohost smoke cases, plus complete original background and padding
readback. FP8 smoke is explicitly not the BF16 target matrix.

After reading every local/subtree/final numerical value, the revised simulator
driver reuses dead arithmetic scratch for separately uploaded original expected
words. A device loop compares every one of the 62,400 received packet words and
122,880 target-weight words. Its returned count includes both examined words and
mismatches. A deliberate one-bit change in the expected input, and another in
the expected weight, must each produce exactly one mismatch on the selected PE
and none elsewhere. Restoring only that expected PE must then produce zero
mismatches everywhere. This is exact comparison, not hashing or sampling; the
physical driver additionally retains direct full packet/bank readback.

The source clock measures input transmission through return of all 48 outputs.
Host initialization, arming, progress copies and retention are separate. A
complete matrix component interval is not a token rate or all-layer latency.

## Bounded diagnosis

Attempts 001–003 preserve compiler type errors; 004 exceeds the SRAM gate by
512 bytes. Two-word output streaming reduces the maximum to 47,968 bytes in
005, including a declared 4,096-byte stack. Later compiled variants require
their own complete 1,262-PE ELF census.

Attempts 005, 006 and 008 reach the 240-second simulator limit without numerical
acceptance. Attempt 007's per-PE diagnostic strings generate too many distinct
program variants and it is cancelled near the 2 GiB limit. Subsequent diagnostics
print only at the source, first root and last worker.

Attempt 009 proves that host upload/arm/start submission finishes at about
1.41 seconds while device initialization markers occur at cycles 419–1,045.
No device arm/start marker appears before its cycle 132,197 timeout. Therefore
the earlier host-side phase names do not establish a kernel communication
deadlock. Later drivers require returned D2H initialization and arming fences.

The revised simulation uses two memcpy channels and omits unused FP8 smoke
payloads. The complete-matrix run has a 900-second ceiling; preparation and
compilation each retain 240 seconds. Memory remains 2 GiB, swap disabled, two
CPU cores, one shared heavy-work lock and bounded logs/files. All failed or
cancelled attempts retain their sources and receipts. Physical staging remains
gated on a normal, nondiagnostic simulator pass and resource release.

Attempt 011 reaches source return and all 1,262 completion counters before the
simulator receives signal 11 during result readback; the host subsequently
aborts. The exact fault cause is not established. Attempts 012 and 013 reject
48,208-byte and 48,144-byte footprints after adding exact payload validation.
Attempt 014's constant-length transmit specialization increases the footprint
to 48,352 bytes, so the smaller shared helper is restored. Removing unused worker
timestamps admits attempt 015 at 48,128 bytes, but a later simulator readback
crash prevents acceptance. Its response size suggests the new scalar validation
report path, which is an inference rather than a proven simulator defect.

Attempt 016 reuses the already captured two-word local result for validation
counts, removes the separate scalar export, and safely pads non-worker target,
scratch and gather regions for whole-rectangle host copies. The complete-worker
SRAM maximum falls to 48,032 bytes including the unchanged stack. Source,
forwarder and padding storage are part of the full PE coverage gate. Original
worker bank formats, capacities, tile coordinates and arithmetic are unchanged.

Attempt 016 explicitly passes every native dot, subtree and all 48 original
matrix outputs. Its maximum FP64 output error is 3.845198079943657e-7 and its
source interval is 11,170 simulator cycles, including possible live host-copy
overlap. All 62,400 input words pass exact device comparison, including the
one-bit negative control and restored zero mismatches. The 900-second run
deadline expires during the separate 122,880-word weight comparison. This is
partial numerical evidence, not a normal-stop or weight-retention acceptance.
The failed attempt and released workstation service are preserved.

Attempt 017 restores completion-based host command release: each PE unblocks
only after its data/control callbacks, teardown and (for the source) full result
return. One subsequent D2H command reads stable counters, replacing repeated
live progress and confirmation snapshots. An early-finished PE's copy can still
overlap another PE's work; no time is subtracted. The unchanged full-run deadline
also bounds a missing completion. Original numerical, exact word comparison,
negative-control and retention gates remain mandatory.

## Complete simulator qualification

Nondiagnostic attempt 017 passes every gate and stops normally. All 1,262
completion records, 960 local two-row results, 960 subtree results, 24 retained
root pairs and 48 final original-matrix outputs pass. Both exact payload checks
pass: 62,400 input words and 122,880 original target-weight words, with exactly
one mismatch for each deliberately corrupted expected word and none after
restoration. Maximum FP64 output error is 3.845198079943657e-7.

The same-source input-to-full-result interval is 9,328 simulator cycles. This
differs from 016's live-copy schedule and is not a physical speedup claim. The
host observes initialization at 223.477 seconds, arming at 276.928 seconds,
complete counters at 344.585 seconds, numerical outputs at 411.637 seconds,
input comparison at 592.048 seconds and weight comparison at 881.872 seconds.
The entire prepare/compile/run service completes in 912.373 seconds; the run
itself stays within its unchanged 900-second ceiling. All 963 images cover
exactly 1,262 PEs; maximum SRAM including stack is 48,048 bytes, with 80 bytes
remaining. The terminal service and free shared lock are independently recorded.

Physical qualification is recorded below. This successful simulator component
alone is not a completed model or a physical inference-rate result.

## Bounded physical artifact admission

Physical attempt 001's SDK compiler job succeeds and releases, but its host
wrapper rejects a 12,170,645-byte archive against the initial 8 MiB component
limit before starting a runtime job. The original failed host-stage receipt is
preserved. The archive has 991 members and 66,841,978 expanded bytes; its 963
application images occupy 65,676,968 bytes. This code replication cost matters
for later whole-wafer loading and is not a weight-storage measurement.

Attempt 002 reuses that exact archive, without another compile job. Every
embedded CSL source matches, every PE is covered once at physical offset
(123,706), and actual maximum SRAM plus stack remains 48,048 bytes. The geometry
profile alone admits a 16 MiB message cap; unrelated components retain 8 MiB.
A no-network recording stub verifies byte-identical protobuf framing against
the installed SDK: one complete original message versus twelve legacy bounded
chunks. This is transport-framing evidence, not proof of server behavior.
Runtime numerical acceptance and normal resource release remain mandatory.

## First physical observation and failed repeated-epoch gate

Attempt 002 passes its first full original-matrix case on WSE-3: all 960 native
two-row results, all 960 subtrees, 48 final outputs, 62,400 input words and
1,262 complete PE counter records pass. The same-source interval is 10,824
physical cycles; maximum FP64 output error is 3.845198079943657e-7. Initialization
is observed after 109.532 seconds; host upload/arm and start/fence intervals for
this case are 0.003143 and 0.001868 seconds, respectively. They are separate
host observations and are not the device matrix interval or a model token rate.

The next case stops progressing during arm or its following returned audit
fence. The 300-second host deadline cancels the job and cleanup verifies release.
Full bank retention, all six cases, replay and normal stop are therefore not
accepted. Fluency or the successful first output cannot replace those gates.
`execution-diagnostic.json` preserves the positive observations with this limit.

Simulator attempt 018 isolates two epochs using synthetic zero-initialized banks
and three trace PEs, without model upload or the full original readback sequence.
Both epochs, all counters and zero matrix outputs pass in 274.515 runtime seconds
and stop normally. This narrows the diagnosis but does not establish the cause
of the physical stall. A fresh physical attempt 003 changes only the SDK host
transport from two channels to one, retaining the frozen nondiagnostic CSL,
original fixture, arithmetic, six-case checks and deadlines. Its successful
qualification follows. No full-model correctness or throughput acceptance
follows from a matrix component.

## Complete physical qualification

`full-bf16-matrix-hw-003` passes all six cases, exact replay, full original bank
and padding retention, and normal shutdown on one WSE-3. The five BF16 calls
each execute the complete original48x5120 matrix. The additional FP8 call is
explicitly a representative cohost smoke test, not a complete original FP8
matrix. Across all calls,11,520 native and11,520 subtree values,288 returned
values,374,400 input words and every PE's callbacks/teardowns pass. Original
resident payload33,832,240bytes and13,520padding bytes are retained exactly;
only the target matrix has complete frozen-atlas slot identity.

Each BF16 call, including replay, measures10,824same-source physical cycles.
The cohost FP8 smoke call measures10,633cycles with different weights/operands.
Mixed-input BF16 maximum FP64 output error is3.845198079943657e-7; the much larger
boundary-normal inputs have maximum0.0503997802734375, within the independently
frozen bounds. Every ordered FP32 output remains exact. No tolerance was changed.
Actual maximum SRAM including4KiB stack is48,048bytes across all1,262PEs.

Host bank upload/initialization takes0.336712seconds after the runtime context
becomes ready; it excludes artifact/runtime startup. Full compile and run
lifecycles are74.501 and121.497seconds, including setup and cleanup. Both jobs
succeed and release, and a fresh account/system audit records no owned
allocation. These lifecycle times are not matrix latency or provider billing.

The one-channel host transport succeeds with unchanged kernel, original fixture
and numerical/retention gates. The exact cause of the earlier two-channel stall
is not established. The first BF16 device interval is10,824cycles in both runs,
so this is an execution-reliability change, not an arithmetic speedup. See
`full-matrix-summary-001.json`. Full-model value production, all-layer/state
feedback and the2,000dependenttokens/s target remain unqualified.
