# Full136 origin word receiver

Physical RUN010 still fails the original strict 650-record criterion. The origin
now receives each IQ data word as a task argument and stores it with scalar
instructions. The same malformed first eight words recur, despite changed
packet ordering and retained RX/TX context. Bulk origin body DMA is not necessary
for this occurrence; no unique upstream cause or repair is claimed.

These two CSL files are exact COMPILE012/RUN010 source. The patch records the
complete change from RUN009. The other four CSL files, nine qualified host files
and runtime adapter reuse exact [first_fault](../first_fault) sources, pinned
in the evidence map. Historical production-packets.csl is not selected. No
vendor SDK, compiled artifact, raw capture, execution guard or admission ships.

Each origin data/control handler blocks queue picking before changing state.
Only valid continuations unblock it; failure retains the RX lease. Native41
still commits the packet after its full body. Host acceptance remains all650
records plus two complete identical zero fault banks. This diagnostic does
not alter producer concurrency, wire framing or application payload criteria.

[Report](../../../../docs/FULL136-WORD-RECEIVER.md) |
[Evidence](../../../../evidence/ready-fanin/native-control/full136-word-receiver.json)
