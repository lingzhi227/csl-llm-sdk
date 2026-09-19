# Native control with two independent sources

Three application PEs share a westward message route: receiver x0 and senders
x1/x2, with x1 also forwarding x2. One device GO starts both senders. Sender1
sends8 then8 words; sender2 sends31 then26, for four packets and73 words. The
same native-control packet source is retained from the earlier qualification.

The independently accepted simulator result checks every payload word, all
complete source/frame/RX banks and retained suffixes, strict per-source order,
released leases, four native control consumptions and a stable final cache.
Each sender observes a tagged first-start notice while its own first-packet
lease remains held. Both witnesses are mandatory. They establish overlap of
issued-to-software-release lease intervals, not simultaneous DMA or router-cycle
arbitration. Notices do not delay tails or gate second packets.

Physical compilation preserves the same three CSL files, task parameters and
traffic. Its three actual programs have identical executable sections, task
tables and DSR instructions to the simulator compile, with different fabric
geometry and ELF files. Physical run005 independently passes the same strict checks, including both
first-packet overlap witnesses, with seven saved captures and normal stop.
The failed136-producer physical fixture remains a separate unresolved result.

layout.csl, pe.csl, packets.csl, driver.py and check_multisource.py are exact
accepted sources. capture.py and capture_store.py adapt the same observations
to an entered physical runtime; the15 host changes are recorded in
PHYSICAL-HOST-DELTA.json against simulator-driver-original.py. capture() requires
an externally admitted physical context. Site allocation and artifact guards
are deliberately not distributed as an immediately submit-ready job.

checker_tests.py exercises six legal arrival interleavings and658 contradictions
without importing the SDK. capture_tests.py is a Linux host-only fake-runtime
fixture covering the full14-copy budget, a returned-copy failure and a stop
failure. It creates synthetic files in host-mock-results and refuses a repeat
in that directory. Neither test is device or original-model acceptance.

[Report](../../../../docs/NATIVE-CONTROL-MULTISOURCE.md) |
[Sources](../../../../evidence/ready-fanin/native-control/multisource-source-map.json) |
[Attempts](../../../../evidence/ready-fanin/native-control/multisource-attempts.json)
