# Concurrent READY framing reproduction

This is the exact six-file device source used by a bounded SDK attempt and a
physical WSE-3 reproduction. The physical attempt failed strict packet checking,
while all 408 producer source snapshots were correct. This is failure evidence,
not a transport pass or neural inference result.

The application is 178 x 2 PEs. Row 0 contains the origin at x0, a phase-2 peer at
x2, and 136 producers at x40..175. All producers send concurrently after one GO.
The peer returns eight synthetic rows in 31/31/26-word fragments. Each active
column independently drains diagnostics into an idle sink on row 1.

Run `python3 -B source_tests.py` for the strict saved-record checker fixtures.
`compile_sdk.py` and `driver_sdk.py` preserve the failed simulator attempt's
actual entry points; a repeated run is not implied. `compile_hw.py` retains the
separate physical compiler workflow and requires a fresh manifest and exact
admission through `source_gate.py`. Set CSL_INSPECT_PYTHON to the site SDK
interpreter supporting cerebras.elf. No execution admission is distributed.

The physical `capture.py`, `capture_store.py` and `capture_tests.py` contain the
actual bounded sink-only capture and durable per-return storage. Capture accepts
an already authorized runtime adapter with a stop method that exits its context;
site allocation and original artifact bindings are intentionally not packaged.
This source is a reproduction component, not a ready-to-submit job.

`production-packets.csl` is an unimported baseline. The imported packets.csl adds
only the pre-SDK observation hook. Source snapshots do not continuously observe
DMA memory. Sender diagnostic overflow is not exported; zero sink errors alone
does not establish zero sender overflow. Check actual sequence contents.

[Report](../../docs/READY-FANIN-REPRODUCTION.md) ·
[Attempt scope](../../evidence/ready-fanin/attempts.json) ·
[Source correlation](../../evidence/ready-fanin/source-correlation.json) ·
[Source provenance](../../evidence/ready-fanin/source-map.json)
