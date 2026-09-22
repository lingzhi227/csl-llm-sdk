# Full136 actual sender-frame point checks

Physical RUN011 still fails the original650-record transport criterion. All136
producers traverse three qualified checks of the actual staged header, payload,
source and native-control tail, yet the same malformed receive prefix recurs.
The checks occur before frame issue, before tail issue, and before source
release. They do not observe continuous DMA or wire data and do not prove a
unique routing cause.

`packets.csl` is exact COMPILE013/RUN011 source. The patch records its full change
from COMPILE012/RUN010. Reuse exact `pe.csl` from [word_receiver](../word_receiver)
and the other14 own source files from [first_fault](../first_fault), pinned in
the evidence map. Historical production-packets.csl is not selected. No vendor
SDK, compiled artifact, raw capture, execution guard or admission ships.

[Report](../../../../docs/FULL136-SENDER-FRAME.md) |
[Evidence](../../../../evidence/ready-fanin/native-control/full136-sender-frame.json)
