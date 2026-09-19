# Small two-source native-control comparison

The same three-PE application tests two independently started senders sharing
one receiver route. The simulator passes strict packet, stable-state and causal
lease-overlap checks. Physical compilation is independently accepted. The physical
runtime also passes all independent checks and has released its system allocation. The separate136-producer physical failure
is preserved, and complete original neural-layer epochs remain zero.

## Matched application and observations

Receiver x0 and senders x1/x2 use the actual SDK message-passing routing bank.
The four packets have lengths8,8 and31,26, with73 source/sequence-tagged words
including full-u32 edge values and values resembling network/control headers.
One device GO starts the traffic. No host pacing, ACK, tail delay or notice-based
second-packet gate is added. The first tail completion records the complete
source/frame banks before immediate source reuse for the second send.

Four complete31-word receiver banks preserve actual arrival order and retained
short-packet suffixes. Each sender retains two31-word sources, two32-word frames
and three-point lease proof bits. Cached48-word states retain phases, errors,
header/length/control scratch, actual native/ordinary-tail and callback counts,
full sender statistics and first-notice fields. Current31-word RX banks remain
readable after a protocol failure even if no delivery callback occurred.
Uncompleted TX proofs are absent completion evidence, not snapshots of live TX.
A stalled host read cannot establish what an unreturned array contained.

Each sender emits one tagged notice after its first asynchronous frame is
issued. The receiver of that notice must still own its own first source/frame
lease. Both exact witnesses131335 and65799, one notice each and no late/duplicate
flags are required. This proves overlapping issued-to-software-release lease
intervals only. It does not show concurrent DMA or same-cycle router arbitration.

## Accepted simulator result

Compile018 failed parsing because maybe_start lacked its final closing brace.
It produced no ELF. Compile019 adds that single brace; the original failed
attempt remains preserved. Source delimiter checks now reject the known defect,
and actual CSL compilation confirms the corrected candidate. All three actual
programs pass full storage, task-table, DSR ownership and fit review at maximum
16,320 bytes including the4,096-byte stack allowance under the48,128-byte gate.

SIM011 saved eight arrays,6,404 stored bytes, with eight D2H calls carrying4,292
host bytes, five launches and zero H2D. All73 words and complete banks passed.
Arrival order was1.0,2.0,1.1,2.1. An initial cached native count of2 progressed
to4; the complete settled state and final snapshot were byte-identical. Both
first-packet witnesses passed. The context stopped normally and the actual
executor exited0. Independent raw checking did not import the submitted checker.

Three SDK-generated ELF identities were first observed, then independently
checked against installed SDK source and geometry. Their hashes were not assumed
from the older two-PE run. Original compiled files stayed unchanged. Resource
release was independently verified; no extra simulator run was used to obtain
hashes or to complete provenance.

## Physical deployment comparison

Physicalcompile007 uses identical CSL/application3x1/offset4,1 on a762x1172 fabric
instead of simulator10x3. Three actual ELFs and all retained banks, task tables,
complete DSR instructions and static fit pass independent review. All executable
sections and task-table bytes/addresses match019. The complete ELF files and
fabric dimensions differ. The physical SDK/compiler/service path is an explicit
combined deployment factor, so a differing outcome would not uniquely identify
silicon behavior.

Physical run005 saved seven arrays, 5,564 stored bytes, using seven D2H calls
carrying 3,716 host bytes, four launches and zero H2D. All four packets and
73 words, full banks/suffixes, exact order1.0,2.0,1.1,2.1, both first-packet
witnesses and complete final stability passed an independent raw audit. The
first physical snapshot was already settled. Six common named arrays match
SIM011 byte for byte; physical state-0 matches SIM011 state-1 and final. Seven
versus eight arrays and different intermediate polling are preserved, so this
is not a claim of identical complete suites or timing.

The actual owner exited0, the context stopped normally, and the correlated job
was terminal with no system assignment or remaining owned processes. Total job
wall time was61.496s; context entry took51.811s and context exit8.174s. These
service/lifecycle observations are not token latency or neural performance.
The seven raw arrays and original receipts were backed up at the compute site;
no raw arrays, compiled ELFs or model weights are in this public snapshot.

The runtime envelope is300s outer,30s without capture progress and90s from
context entry through exit. The maximum is14 D2H calls,7,748 host bytes,
11 launches, zero H2D,16 raw files of at most4,096 bytes and1MiB aggregate.
Every returned array is durably stored before its checks or subsequent reads.
Host-only tests cover full capacity, a failed fourth copy and a failed stop;
each attempts stop once and preserves prior data. Actual owner reap and an
independent terminal/unassigned/process-absent release are separate requirements.

This physical pass qualifies only the small matched case. It does not separate
the large fixture's sender count, route distance, bidirectional request lifecycle
or packet mix. A protocol pass
without both overlap witnesses would remain protocol-only. A host timeout would
be an incomplete observation, not an inferred protocol error. No unchanged
case is retried automatically.

[Exact own source](../examples/ready_fanin/native_control/multisource) |
[Attempt receipts](../evidence/ready-fanin/native-control/multisource-attempts.json) |
[Source hashes](../evidence/ready-fanin/native-control/multisource-source-map.json) |
[Preserved physical136 failure](NATIVE-CONTROL-PHYSICAL.md)
