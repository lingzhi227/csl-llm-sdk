# Complete class lowering and composed bank engine

Full-model performance is still unqualified. This work binds all original
projection descriptors to the P15 storage classes and develops one executable
input/compute/reduction/return engine. Its bounded execution is a twelve-worker
component, not a complete original K width, matrix or model.

## Complete original-model address and forest binding

`coupled-plan-002` composes the hashed original P12 atlas with the hashed P15
columnar overlay. Inherited state/KV/value/canonical assignments are unchanged.
Projection bundles explicitly name their storage class, and projections from
different physical classes cannot fuse. All 498 original matrices and
105,052,160 tiles still form 306 bundles/980 events; 18 memory completion edges
protect the 192,736 reused-storage value pairs.

The K136 forest now follows its actual 748-column, 857,208-bank subring. Other
FP8 forests follow the main ring, and BF16 dispatch applies phase 480,000 over
its unchanged state-excluded ring. Whole-K alignment and actual physical group
roots are checked for every matrix segment, including slot wraps.

`coupled-audit-002` independently expands every tree edge and checks 44,718,648
PE/color entries, the compact patterns, physical bounds, bridge coordinates,
class eligibility, every original matrix dispatch and graph dependencies. It
also matches all FP8 input consumers against actual reduction participants and
checks concurrent input/reduction color disjointness. Dense routes remain on
the workstation; published evidence contains hashes and compact metadata.
The audit completes in 3.613 seconds at 241,900 KiB peak RSS under the bounded
2 GiB/no-swap/two-CPU service. All resources are released.

Plan/audit 001 remain frozen. Version 002 corrects inherited metadata: the old
base bank census is omitted, resident matrix bytes are 29,840,350,720 and segment
descriptors are 619. Overlay integration obligations are included. All graph
events, dispatch descriptors, profile records and dense forest hashes are
identical between versions; only the bound materialized-atlas identity changes.
The complete bank census is independently recorded in `columnar-audit-002`.
No physical code changes or extra physical allocation are needed for this fix.

| Profile | Contributing PEs | Trees | Maximum single edge hops |
| --- | ---: | ---: | ---: |
| FP8 K40 | 860,880 | 21,522 | 22 |
| FP8 K48 | 860,880 | 17,935 | 26 |
| FP8 K136 | 857,208 | 6,303 | 70 |
| BF16 K40 | 824,000 | 20,600 | 797 |

These are abstract routing/address checks, not physical full-matrix execution.
BF16 inputs and global result return are not admitted by this audit.

## Executable composition

The 4x5 component preserves P15's horizontal 130-word pair selection and vertical
65-word operand selection. Twelve original mixed-bank workers then execute native
FP8/BF16 dots, join a fixed ordered tree, and return a three-word result to the
source controller. The return includes two FP32 values and an exact participant
count. Every worker subtree has an independent ordered FP32 and FP64 oracle.
The source controller measures its own send-to-result-return interval, avoiding
cross-PE clock subtraction.

| Bank operation | Queue | Microthread | Descriptor register |
| --- | --- | --- | --- |
| Filtered input | IQ2 | 2 | 2 |
| Left child | IQ3 | 3 | 3 |
| Right child | IQ4 | 4 | 5 |
| Parent/root return | OQ2 | 5 | 6 |
| Native dot | Local memory | Synchronous | 4; BF16 also 7 |

The source and relay roles separately use OQ2/UT3, and the source return uses
IQ3/UT4. Child receive buffers are distinct from decoded-weight scratch. Child
receives are armed after the local native dot; early child traffic is subject
to ordinary queue backpressure. The outgoing aggregate reuses dead decoded
storage. Full original bank/padding readback and input-packet checks remain.

A same-color hardware router is not an arbitrary merge point: simultaneous
arrivals from enabled receive directions are undefined in the
[SDK routing semantics](https://cerebras-sdk-docs-130.netlify.app/csl/language/builtins#routing-configuration-semantics).
This engine gives every tree edge a unique active PE/color path. The single
root return does not establish a many-root whole-matrix output collector.

Attempts 001/002 preserve a composition failure: four two-child joins use stale
scratch contents although final child-count readbacks are correct. Attempt 002
records the counts and fails the unchanged host gate. Independent child DMA
buffers remove the observed failure in 003, which passes all simulator numerical,
retention and shutdown gates. This isolates a problematic shared-buffer use but
does not prove a compiler defect or a precise hardware ordering root cause.

## No-temporary FP8 decoder

The original finite E4M3FN-to-half/256 bit embedding is
`E(c) = ((c & 127) << 7) | ((c & 128) << 8)`.
For packed low/high bytes, sign-propagating shifts and masks implement the same
embedding in five vector operations instead of nine, without the old 256-byte
temporary array. The new library uses the documented native
[`@sar16` arithmetic shift](https://cerebras-sdk-docs-130.netlify.app/csl/language/builtins#sar16).
The original decoder is retained as the comparison implementation.

`fp8-shift-proof-001` exhaustively checks all 65,536 packed patterns against the
original integer map and independently checks all 254 finite codes against IEEE
half encoding of the mathematical value divided by 256. It checks signed zero
bits as well. Invalid FP8 NaN encodings are included only in bit-map identity;
the original finite-value admission remains unchanged.

Four passive probe PEs each enumerate a disjoint 16,384-pattern range and compare
both device decoders element by element. They measure matched decoder intervals
on the same local clock. Original native-dot and full composed-tree gates are
also required after this test. Pure-host bit identity alone does not qualify the
device instruction or model performance.

## Physical qualification and measurements

`coupled-bank-hw-001` passes all ten mixed/zero/tiny/boundary/replay cases:
240 native values, 240 subtree values, 20 returned values, 7,800 worker packet
words and 3,900 relay pair words. Every expected callback/teardown and participant
count matches. Original payload readback covers 421,560 bytes; the unified
423,072-byte transfer also checks all 1,512 padding bytes. Numerical comparisons
use exact ordered FP32 results with canonicalized zero signs and independently
frozen FP64 bounds. Normal shutdown completes.

The exhaustive physical decoder comparison reports zero mismatches over all
65,536 packed patterns. Matched same-PE intervals sum to 250,563 cycles for the
original decoder and 130,498 for the shift decoder: **1.92005x** local speedup.
These intervals include the same measurement overhead; they are not a paired
whole-engine or full-model speedup. Both decoders remain in the qualification
artifact, while all native dots use the new decoder.

| Component case | Source send to controller result return, cycles |
| --- | ---: |
| FP8, 260-word input stream, mixed/replay | 1,685 |
| FP8, 780-word input stream, zero/tiny/boundary | 1,906–1,907 |
| BF16, 260-word input stream, mixed/replay | 1,915 |
| BF16, 780-word input stream, zero/tiny/boundary | 2,499 |

The overall ten-case median is 1,911 cycles. The source's local clock includes
two-stage input transport, native arithmetic, ordered tree and result return.
Host upload/arm, final send callbacks and retention checks are excluded.
There is no assumed clock frequency or conversion to model tokens/s.

Maximum compiled SRAM plus the declared 4,096-byte stack is **48,032 bytes**,
leaving only **96 bytes** below the unchanged 48,128-byte application ceiling.
This includes the decoder comparison hooks and both native precisions. It does
not admit a generic forest scheduler or the remaining complete-model roles.
Simulator 003 used the old decoder at 48,112 bytes. Simulators 004 and 005 pass
the exhaustive comparison and all composed gates at 48,032 bytes; changing an
unused reference configuration in 005 produces no further memory improvement.
The final simulator completes in 205.996 seconds under the bounded service.

Compile `wsjob-ugkmzdlbkwtonrtujqnnp6` and runtime
`wsjob-b6ssphgatthacercjs3ody` both succeed and release normally, with stage wall
times 71.480 and 81.453 seconds. A fresh job/system snapshot at
2026-09-27T17:42:00Z confirms no owned active jobs or system assignments.
These wall times are not provider billing. All workstation services and the
shared lock are released. Artifact SHA-256 is
`e26e6b0f496760ee5d043503780bad252e901c5df3408a2ab1f4435f90b19265`.
See `coupled-bank-summary-001.json`, the frozen physical receipts and all five
preserved simulator attempts. The source suite passes 39 tests.

## Remaining integration

The complete plan and the bounded engine must still be combined into actual
full-sized matrix execution. Remaining work includes real quant/value producers,
BF16 operand distribution, bidirectional vertical input, many-root output
collection and arena writes, generic route transitions and all-layer control.
Combined GDN/state/attention roles and full scheduler memory need their own
compiled admission. Whole-matrix numerical checks and measured complete model
TTFT/dependent decode remain mandatory; the 2,000 tokens/s goal is unmet.
