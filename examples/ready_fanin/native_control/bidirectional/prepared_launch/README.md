# Prepared-launch bidirectional physical pass

This is the exact PE source from independently accepted physical RUN007.
Use it as pe.csl with the unchanged layout.csl, packets.csl, row_packet.csl and
host/checker sources in the parent directory. The parent pe.csl remains the
original failed RUN006 source. Site-specific admission, bindings and guards
are required separately; these source files do not submit a device job.

The only PE changes move the existing one-shot GO from compute into origin's
pump, after row and REQUEST0 preparation and just before send_packet, for
group0 only. Producer preparation, traffic, callbacks and strict checks are
unchanged. SOURCE-DELTA.json is the original proposal record; its historical
unexecuted wording is superseded by the linked acceptance evidence.

RUN007 passes all 31 checks: 9 packets/200 words, two assembled and two released
source rows, full banks and suffixes, stable state, and both first-TX lease
witnesses. Origin separately records RX during an unfinished TX. This proves
the observed software lease overlap, not simultaneous DMA. One pass does not
identify a unique root cause or repair the larger136-producer failure.

[Report](../../../../../docs/NATIVE-CONTROL-BIDIRECTIONAL.md) |
[Accepted evidence](../../../../../evidence/ready-fanin/native-control/bidirectional-prepared-launch.json)
