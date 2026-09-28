# Shared input for full and ragged native owners

P30 targets duplicate traffic between quantized producers and native MLP
consumers. Original5120 ->17408 ->5120 dimensions, weights, input operands,
projection arithmetic, scales and BF16 boundaries remain unchanged. The
enclosing mixer/state operations are inactive. This component is not a complete
neural layer or model-throughput result.

## One wire copy, both original owners

P29 sends each native K slice twice: once for full native owner groups, then
again under a different selector for the short final owner group. Both copies
carry identical operands and scales. Only the selector and local part differ.

The new optional lowering transposes selector numbering. For B original K
slices and T tail workers, define S=ceil(B/T). Slice k uses selector
base+(k mod T)*S+floor(k/T). A full-group worker filters its exact selector;
tail worker r filters the contiguous range from base+r*S through its last
actual part. Every word retains the hardware range-filter selector in its
upper half. The lower header carries the tail part. Single-slice native owners
store that operand at local part0; multi-slice owners retain the transmitted
part. Unused range entries are excluded, including uneven final ownership.

This shares the exact same wavelets with both consumer classes. It halves
native broadcast payload from49,376 to24,688 words and native slice frames
from864 to432. Each group still uses one output DMA:176 coalesced packets.
The largest packet shrinks from296 to148 words. The40 preparation commands,
186 response grants, independent273-word response buffer, one-frame prefetch
and all completion/drain rules remain intact. Cardinal routes and original
weight locations are unchanged; only input RAMP filter ranges change.

Independent checks reconstruct every original PE's accepted native operands
and scales for both stage geometries. They compare shared delivery with the
old full/tail delivery under shuffled arrival, reject missing/duplicate parts,
and reject corrupted selector ranges, part headers and buffer extents. All134
source tests pass. Static traffic savings do not establish elapsed-time savings.
Fresh full-stage and physical qualification results follow below.

## Native-loop alternatives retained

A matched eight-PE simulator probe first checked two native-loop alternatives.
Both keep original resident banks, ascending-K FP32 accumulation, all scaling
boundaries, all eight inputs/twelve timing modes and independent FP64 bounds.
Both numerical runs pass exact outputs, packet/counter checks, bank retention
and normal stop. These are simulator results only.

| Used shape | Baseline decode + dot | Paired-input callback | Fully unrolled input |
| --- | ---: | ---: | ---: |
| 4x64 | 572.9375 | 574.9375 | 545.84375 |
| 8x32 | 553.9375 | 556.9375 | 526.90625 |

Values are simulator counter means over32 local invocations. Pairing is slower
on both used shapes. Unrolling reduces the local counts by4.73% and4.88%, but
the complete11,388-PE layout then fails SRAM admission on68 PEs: maximum48,320
bytes including the unchanged4,096 stack,192 bytes over the48,128 ceiling.
Neither alternative replaces the qualified kernel or receives a WSE trial.
The first unroll source's invalid comptime loop is retained as failed004;
explicit constant-index statements pass simulation005 before full compile008
rejects capacity. This avoids selecting a local speed result that cannot fit
the actual fused program.

Sources and numerical receipts are frozen under native-shapes-sim-003..005,
and the rejected complete census under layer-mlp-compile-008. The shared-input
candidate keeps the original native kernel. Residual/norm, adjacent full layers,
isolated state and complete64-layer sentence throughput remain unfinished.

## Complete physical qualification

Physical006 uses exactly the CSL admitted by local full-stage compile009.
The physical artifact covers11,388 PEs with4,179 ELF images. Its maximum is
48,128 bytes including the4,096 stack reserve, versus48,112 in the local compile;
the physical result controls admission. Three PEs retain zero margin.
All four frozen original-MLP cases,20,480 exact BF16 outputs, native ingress and
scales, counters, all original banks/tables, all-PE drain and normal stop pass.

| Case | P29 ticks | P30 ticks | P29 completed output | P30 completed output |
| --- | ---: | ---: | ---: | ---: |
| Normalized nonzero | 613,188 | 564,333 | 2.969ms | 2.834ms |
| Zero after nonzero | 577,320 | 529,430 | 2.760ms | 2.633ms |
| Changed nonzero | 613,127 | 564,289 | 2.727ms | 2.580ms |
| Warm replay | 613,190 | 564,307 | 2.673ms | 2.551ms |

Nonzero controller counts decrease7.965–7.972%, and zero decreases8.295%.
The normalized phase counts are108,644 /311,470 /144,219. Neural work overlaps
these intervals, so none is labeled isolated transport time. Halving wire words
has not halved the complete component's counters. The four host observations
are shorter than P29 but still longer than P28; no sustained wall speedup or
model-token rate is claimed. Transport counters issue/retire176 packets and
issue176 grants during packet leases, with zero responses arriving during those
leases. The limited observed prefetch overlap remains.

SDK loading takes418.693s, bank/setup initialization3.003s and full diagnostic
runtime440.328s, including7.211s normal stop. Actual neural output timing is
reported separately above. The54,531,355-byte artifact has SHA-256
`e1dc1dd50e23fbebb6ce6197f068936d63efdcde51b91df41f06428555c93054`.
Exact executed source, oracle, results and remote raw array/census hashes are
bound by shared-input-milestone-001.json and the immutable attempt receipts.

Both hardware jobs succeed/release. The final account snapshot reports no owned
active job or assigned system. All five bounded workstation services are released
and their shared lock is free. New workstation experiments retain598,069,960
unique bytes; the complete retained MLP hardware family occupies2,104,208,432
unique bytes. Original payloads are hardlinked. Provider-billed node-hours are
unavailable; SSH connection state is not used as evidence of resource release.

All134 tests pass in both the source and publication trees. The original
functional release and all prior failed/accepted runs remain preserved. Next
work connects actual residual/norm and neighboring complete stages with device
arrival-driven handoff. The complete original64-layer model, correct dependent
sentences, sufficient request-state capacity and>=2000 aggregate generated-token/s
acceptance remain outstanding.
