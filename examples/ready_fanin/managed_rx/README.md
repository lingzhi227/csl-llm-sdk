# Compiler-managed receive descriptors

This exact variant compiled for all 356 fixture PEs, then failed on physical
hardware. It retains the combined TX buffer and operation. RX uses DSD operands
instead of explicit DSR6, retaining UT7, callbacks, destination buffers, lengths
and leases. The actual origin, peer and producer programs use RX register pair1,
TX pair2 and synchronous copy pair3; this is scoped static allocation evidence.

To assemble this variant, copy the parent ready_fanin example to a new directory,
first overlay packets.csl, TX-DELTA.json and source_tests.py from combined_tx,
then overlay this directory's packets.csl, RX-DELTA.json and source_tests.py.
Run the resulting source_tests.py. Other five CSL files and the strict checker
are byte-identical. A fresh manifest and site-authorized compile/runtime are
required; historical admissions and compiled artifacts are not distributed.

The maximum actual fixture storage including 4 KiB stack remains 25,600 bytes.
The new receive allocation did not repair the protocol: all 13 returned captures
are byte-identical to combined-TX runtime002 on the same physical system. The
first malformed payload remains [10,1,1,0x00400108,0,10,0,1] at origin record22.
All 408 producer records were independently exact. Normal exit and resource
release do not make this a protocol pass. No full-model promotion is claimed.

[Report](../../../docs/READY-FANIN-REPRODUCTION.md) ·
[Provenance](../../../evidence/ready-fanin/managed-rx-source-map.json) ·
[Capture receipts](../../../evidence/ready-fanin/managed-rx-capture-receipts.json)
