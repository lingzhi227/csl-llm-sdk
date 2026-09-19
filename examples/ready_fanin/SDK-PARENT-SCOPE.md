# Finite same-row READY fan-in discrimination

One round, 136 concurrent READY producers at application x40..175/y0; origin x0,
actual phase2 postnorm peer x2, physical offset (4,1). Owner122 remains x162.
The 178x2 application has a dedicated idle diagnostic sink immediately below
each active source. A GO broadcast releases every producer together, joined with
its compute arrival. No serialization or per-producer grant is introduced.
The origin concurrently sends eight actual eight-word REQUEST controls to x2.
The unchanged production row_packet.csl emits 31/31/26-word fragments per row,
with 128 finite, group-distinct BF16 bit markers 0x3f00 + group*128 + index.
The peer keeps at most one pending request until the previous source lease is
released. The origin validates exact CHUNK envelope and packed markers before
calling the unchanged row receiver. Strict READY reserved-word zero remains.

The exact fullgraph002 production packets.csl is retained as production-packets.csl.
The sole active transport change is a callback immediately before the SDK send
call. Enqueue and source-completion snapshots occur in the fixture caller.
Packet send/RX ordering, length guards, DSR6/UT7 header and payload receive,
deferral and source leases are unchanged. Instrumentation can affect scheduling;
a passing fixture does not prove the original physical fan-in cause or cure.

Each source snapshots its first eight application words at enqueue, before SDK
and source completion. These are three point observations, not proof of continuous
DMA immutability. Each origin receive records its raw network header, actual
length, first eight application words and rejection reason. First erroneous
records are never overwritten. Producer logs are 3x16 words; peer and origin
capacities are 96 and 192 records. Expected counts are 3 / 81 / 161 respectively.
Sender overflow increments a local counter and drops further records. Sink
status observes sink overflow only: sender overflow is not exported. A saturated
failure prefix without terminal evidence is potentially truncated; do not claim
no sender overflow. There is no reset or autoregressive acceptance in this round.

Diagnostic append performs bounded memory stores and activates task18. It does
not synchronously send or block the packet callback. Task18 drains each immutable
record asynchronously to its dedicated vertical sink. Per-record source memory
remains owned through task14 completion. Sink-only publication uses production
observer-style ordered odd/body/even boundary words, with no record reuse.

Resources: packet IQ/OQ6, RX DSRdest/src1=6 and UT7, SDK TX default UT6;
packet local tasks12,13,15,16. Diagnostic IQ/OQ5, DSRdest/src1=7, UT5, tasks14/18,
color5 vertical routes. GO OQ/IQ3, data task3, explicit synchronous destDSR4,
color3 broadcast. Fixture service task17. SDK memcpy queue/UT/DSR0 and command
queue1 remain separate. Actual implicit compiler DSR allocation is not claimed.
Normal stop and owned cgroup release are mandatory, even with idle RX armed.

Host launches initialize and compute once each. Initial state covers356PEs;
after compute, reads target only row1 diagnostic sinks. There are at most32
status polls with at most25ms host spacing, followed by three record rectangles
(origin3072,peer1536,136producer x48 u32) and one final status read. All returned
copies are durably saved before the next SDK call. Minimum6, maximum37 copies;
maximum returned array payload238,208 bytes before NPZ overhead; at most37 raw files.
This keeps the known SDK long-idle fresh-D2H issue explicit rather than silently
patching vendor code. An actual failure is preserved, not automatically retried.

Proposed one-attempt workstation envelope: compiler300s, driver180s, whole owned
unit520s; CPU0, aggregate cgroup memory4GiB, per-child AS4GiB, Swap0, tasks128;
file16MiB, log4MiB, candidate256MiB, global RAM reserve8GiB, SSD/HDD reserve32GiB.
Heavy lock and real cgroup/affinity/pressure checks are inherited. All artifacts
and raw capture remain on Mass1. No model weights, SDK binary patch, physical
allocation or full original neural inference is involved. No execution until
exact frozen manifest/profile controller admission. An SDK pass alone cannot
establish correctness of physical route merging; later physical qualification
would require a separate reviewed attempt.
