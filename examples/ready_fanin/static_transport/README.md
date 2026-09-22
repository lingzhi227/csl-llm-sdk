# Small static retained-packet transport

The accepted SDK fixture has 3 producers and 27 PEs. Both complete raw export
sets pass the static-v1 schema and match exactly, with normal runtime exit.
This source is exact accepted SIM002; layout defaults retain the qualification
gate. Full 136-producer physical transport and original neural inference are unqualified.

Reuse exact `diagnostics.csl` and `row_packet.csl` from
[first_fault](../native_control/first_fault), alongside the five CSL files here.
The evidence map pins all ten own sources, including schema, capture and driver.
The small compile uses fabric 16 x 5,offset 4, 1,memcpy,one channel and one compiler
worker. Host admission, resource guards, binaries, vendor SDK and raw evidence
are excluded; this directory is source evidence, not a managed execution tool.

The retained-input gate delays one completion callback until both full inputs
are held. It does not prove a physical fabric stall or continuous DMA behavior.
The first 45-second runtime timed out during its second read and remains failed.
SIM002 changes only runtime time limits to 75/90 seconds; protocol code and two
complete reads are unchanged. All prior native-control failures are preserved.

[Report](../../../docs/STATIC-TRANSPORT-SMALL.md) |
[Evidence](../../../evidence/ready-fanin/static-transport-small.json)
