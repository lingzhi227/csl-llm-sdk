# Paired column input and maximum resident-bank execution

The complete model still has no qualified performance implementation. This
milestone introduces a complete storage/input-route candidate and a bounded
executable input component. Neither establishes full-matrix or model throughput.

## Complete overlay

`columnar-plan-001.json` inherits the pinned P12 atlas and hashes the unchanged
state, KV, canonical tensor, value arena, normalization and actor assignments.
Every original matrix identity, shape and tile remains present: 498 matrices,
105,052,160 real tiles. The overlay splits FP8 storage into two classes:

| Class | PE count | Tile bytes | Placement |
| --- | ---: | ---: | --- |
| FP8 K40/K48 | 860,880 | 260 | Original bank ring |
| FP8 K136 | 857,208 | 260 | First 1,146 bank rows, columns 1 through 748 |
| BF16 | 824,000 | 512 | Original eligible ring, phase 480,000 |

The wide-class ring makes every column's input blocks form a mirrored pair.
Its local slots follow that PE's complete main-class slot reservation. BF16's
phase avoids colocating 111 FP8 and 13 BF16 slots. The largest bank payload stays
35,256 bytes (110 FP8 plus 13 BF16); 111 plus 12 uses 35,004 bytes. Total reserved
matrix storage is 29,840,350,720 bytes, including alignment padding. This is a
payload census, not whole-model compiled SRAM admission.

`columnar-audit-002` independently enumerates all real tile addresses, occupancy,
slot boundaries and original matrix identities. It also checks every input
consumer against its actual class rank modulo K, not against the lowering's
claimed input block. The full audit checks 8,700,000 dense PE/color entries and
all physical neighbor/endpoint continuity. Dense arrays remain on workstation;
only source, hashes and compact receipts are published. The audit took 9.307
seconds with 528,744 KiB peak RSS under a 2 GiB, no-swap, two-CPU limit. Attempt
001's horizontal dead-end failure remains preserved; 002 terminates each stream
at its last actual ingress.

## Two levels of hardware windows

An actor-row stream contains pairs of encoded 128-value activation packets. Each
packet is 65 words: 64 packed data words and one FP32 scale. A column ingress's
counter filter accepts exactly one 130-word pair. That ingress then sends the
pair vertically; each bank's counter filter accepts the one 65-word packet its
matrix tile needs. Other wavelets continue through the router without a local
software receive/discard loop. The relay can begin vertical transmission as soon
as its pair arrives, while later horizontal pairs are still in flight.

| K blocks | Horizontal maximum words | Vertical colors | Per-column total words | Free application colors |
| --- | ---: | --- | ---: | --- |
| 40 | 260 | 19,20 | 260 | 12 through 18 |
| 48 | 260 | 17 through 20 | 520 | 13 through 16 |
| 136 | 780 | 20 | 130 | 16 through 19 |

Horizontal color 2 is reused on disjoint actor rows. The table reserves the
existing P9 reduction color counts, but does not qualify simultaneous arithmetic
and all routes. Counts are payload loads, not timing or achieved bandwidth.
The full abstract vertical route fans both north and south from each ingress.

`counter_window.csl` uses the installed SDK's WSE-3 compatibility setters, in
order: `set_max_counter(count-1)`, `set_active_limit(period-1)`, then
`set_counter((period-offset)%period)`. Compiler-assigned filters count data and
exclude control wavelets. The native half-wavelet representation is left to the
SDK; the component checks exact delivered words, rather than assuming an
interpretation of undocumented raw counter fields.

The component has one horizontal source, three pair selectors and twelve bank
workers in a 4x5 rectangle. It uses colors 2 and 20, IQ2/UT2 for receive and
OQ2/UT3 for transmit. Per-color teardown callbacks compose with the SDK memcpy
dispatcher. Reconfiguration waits for each role's receive/compute/send completion
and relevant teardown callbacks. Data and teardown share their output queue and
microthread. Host arming between cases is explicit; this is not autonomous
neural feedback.

The twelve workers hold actual original pinned checkpoint tiles, reused with
explicit provenance from the previously frozen mixed-bank fixture. Half exercise
110 FP8/13 BF16 slots; half exercise 111/12. The additional thirteenth BF16 tile
comes from another recorded original bank, with no synthetic replacement.
For bounded host transport, each worker uses one aligned 35,256-byte bank view;
the shorter 111/12 profile includes 252 zero padding bytes that are also checked
on readback. Thus the full transfer is 423,072 bytes, comprising 421,560 original
payload bytes and 1,512 padding bytes. The original native FP8/BF16 kernels are
unchanged. Independent FP32 ordered
oracles, FP64 error bounds, exact packet/relay contents, complete original-bank
readback, callbacks and normal stop form the acceptance gate.

## Preserved simulator qualification

`filtered-bank-sim-004` completes all gates and normal shutdown in 132.991 seconds
under the unchanged 240-second per-phase limit, 2 GiB memory cap and no swap.
Its four cases cover both arithmetic types, both horizontal periods, every
horizontal pair offset from 0 through 650, both vertical packet offsets, and
first/last bank slots. The physical fixture additionally includes zero/tiny
cases and two reset/replay cases. Maximum compiled SRAM including the declared
4,096-byte stack is 47,328 bytes, leaving 800 bytes under the existing ceiling.
This admits only the compiled component, not a general full-model bank engine.

Attempt 001 passed all four packet/arithmetic cases but reached the runtime cap
during full-bank readback and is not a completed run. The replacement combines
separate bank transfers into one contiguous rectangle while retaining every
original byte and checking explicit padding. Attempts 002/003 preserve compiler
errors for global pointer views; 004 forms typed views at runtime. Numerical
criteria, original payloads and runtime/memory limits were not relaxed. All
workstation service/lock releases are recorded.

## Physical qualification

`filtered-bank-hw-001` runs the exact simulator-qualified CSL on one WSE-3 in a
4x5 application rectangle. All ten mixed/zero/tiny/boundary and replay cases
pass: 240 native dot values match the independent ordered FP32 reference and
FP64 bounds, 7,800 worker packet words and 3,900 relay words match exactly, and
all receive/send/teardown/completion counters agree. The full original 421,560
bank bytes plus 1,512 padding bytes survive without reupload. Both first-slot
replays match exactly after precision and filter-window changes.

Local receive-callback to native-dot completion measures 1,026 cycles for FP8
and 992 for BF16 (60 samples each). These use different operands/arithmetic and
exclude upstream input delivery, reductions and host arming. They establish
neither a precision speedup nor end-to-end projection/model throughput. Device
clock frequency is not assumed. Actual physical SRAM plus declared stack remains
47,328 bytes, leaving 800 bytes.

Compile `wsjob-8fim8regkyxeunurvgouuq` and runtime
`wsjob-vcpqesxdu7csnhu8amck9n` succeed and release normally (stage wall times
41.434 and 61.436 seconds). The independent job/system audit at
2026-09-27T17:02:02Z finds zero own active jobs or system assignments; other users'
allocation is untouched. Workstation units and shared lock are also released.
The result, original fixture provenance, actual artifact hash, failure snapshots
and release receipts are preserved. See `filtered-bank-summary-001.json`.

## Integration still required

The overlay is a candidate successor, not an in-place edit of the frozen P12/P14
plans. P14 projection descriptors and reduction forests must be regenerated for
its changed classes and BF16 phase; the K136 path must be lowered over its new
748-column subring. Input sources still need routes from the actual distributed
quant/value owners. BF16 input distribution, result scatter, general route epoch
readiness, bidirectional vertical execution and combined neural/actor SRAM also
remain unqualified. Only then can full-matrix and all-layer integration support
a complete dependent-token measurement. The 2,000 tokens/s goal remains unmet.
