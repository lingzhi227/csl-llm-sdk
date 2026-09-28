# Coalesced MLP distribution and one-frame prefetch

P29 changes the transport between the original input/fused-activation producers
and native projection consumers. It preserves the complete original layer0 MLP,
its four-case independent oracle, native FMA order, BF16 boundaries, resident
weight placement and routing. The enclosing mixer/state operations are inactive.
This component does not qualify a complete neural layer or model throughput.

## Lowered packets and buffer ownership

The spatial network lowering now exposes every distribution packet and slice:
producer group, native K selector, full/tail alias, part, column count and word
offset. An independent audit compares this representation with every native
consumer selector. A wire-equivalence test checks the coalesced packet against
the previous scalar format, including ragged K ownership.

Each complete128-value response now becomes one immutable output packet.
Initial groups contain eight37-word slices; fused groups contain four69-word
slices. This reduces controller send completions from864 to176 per epoch while
preserving all49,376 wire words and their order. It reduces software scheduling
work; it does not compress operands or reduce the static route traffic.

The controller owns separate273-word response and296-word packet buffers.
Only one response grant and one packet DMA can be outstanding. After copying a
complete response into the packet buffer, the controller can grant the next
producer while the previous packet remains leased to output DMA. A newly arrived
native response waits until that packet lease retires. Output segments can copy
directly into the final output array while the last native packet is outstanding.
The grant buffer has its own completion lease.

Completion requires all186 responses consumed, all176 packets retired and all
frame/grant/packet leases empty. Sender queue flush still precedes restoration
of its return-bus receive route. The forty prepare-only quantizer commands remain
inside the controller measurement window; they confer no injection permission.
Queue colors, DSR assignments and microthread IDs are unchanged.

Four physical transport counters record issued packets, retired packets, grants
issued during a packet lease and responses arriving during a packet lease.
The last two demonstrate software ownership overlap only; they do not establish
simultaneous instruction issue or isolated network bandwidth.

## Admission and measurement

An independent event model exhaustively explores the small protocol state space,
tests delayed final packet retirement, and runs randomized full176-packet,
186-response schedules across warm epochs. It rejects premature grants, buffer
reuse and rearming. All132 source tests pass. Fresh physical compilation covers
11,388 PEs with4,179 ELF images; maximum SRAM including the unchanged4,096-byte
stack reserve is48,128 bytes. Three PEs have zero remaining margin.

The four controller timestamps retain the same semantic boundaries: start,
last initial packet completion, last fused packet completion, final output and
controller drain. Native computation overlaps those intervals. The explicit
host timer includes arm through blocking receipt of all5120 outputs; SDK loading,
bank initialization and diagnostic validation remain separate. Comparisons use
the accepted P28 physical004 and the identical frozen oracle. Per-case wall
observations and raw counter ratios do not establish sustained model throughput.

## Physical observations and resource release

Physical005 passes all four original warm cases and20,480 exact BF16 output
values, all native ingress/scales, worker/sender/transport counters, original-bank
and table retention, all-PE drain and normal stop.

| Case | P28 ticks | P29 ticks | Fewer ticks | P28 completed output | P29 completed output |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normalized nonzero | 713,706 | 613,188 | 14.084% | 2.801ms | 2.969ms |
| Zero after nonzero | 683,492 | 577,320 | 15.534% | 2.521ms | 2.760ms |
| Changed nonzero | 713,752 | 613,127 | 14.098% | 2.555ms | 2.727ms |
| Warm replay | 713,625 | 613,190 | 14.074% | 2.388ms | 2.673ms |

Every observed host completed-output interval is slower than P28. These four
single observations do not isolate host/SDK variation or establish a sustained
wall-time regression, but they cannot support a wall-speedup claim. The device
counter reduction is limited to the measured complete MLP window. No calibrated
frequency or model rate is inferred.

Every case issues and retires176 packets and grants176 responses while the
outgoing packet lease is live. Zero responses arrive before that lease retires.
Thus the prefetch path executes, but does not hide a complete next response
under the current broadcast in these cases. Coalescing and early grants changed
together; this experiment does not attribute the counter reduction to each one.
The normalized case phase counts are122,885 /344,874 /145,429. The last interval
is slightly larger than P28's144,179; the middle interval still includes native
projection/activation work and cannot be labeled pure communication overhead.

SDK loading takes321.592s, original-bank/setup initialization3.022s and the
complete diagnostic340.338s, including3.793s normal stop. The54,495,659-byte
artifact has SHA-256
`8b17de8f8804e9ad27d0feef0c34e8f68a0d2cea65215f8a349afce9924cf5df`.
The bound milestone and comparison receipts retain exact source/result hashes.
Original actual arrays and the full ELF census remain remote.

Both compiler and runtime jobs succeed and release. The final account audit
finds no owned active job or assigned system; the workstation has no active
service and the shared heavy-job lock is free. No workstation heavy job was
needed. Immutable payload hardlinks avoid retransferring the original banks.
The retained MLP family occupies1,744,139,701 unique file bytes on ALCF; provider
billed node-hours are unavailable from this accounting interface.

Physical004 and005 are retained as numerical and timing baselines. Next work
must reduce actual native/distribution service and remove centralized transfers
where measured benefit justifies the route/resource change. Residual/norm,
neighboring layers and the complete original64-layer sentence and>=2000 aggregate
generated-token/s target remain open.
