# Full136 first-fault physical diagnostic

These are the exact own device and capture sources used by physical RUN008.
The original strict 650-record transport criterion FAILED. Independent review
accepted the failure evidence, completed scalar snapshot, normal context exit,
resource release and durable backup. This is not a transport repair or neural
inference pass. The original RUN004 and later small-fixture histories remain.

The six CSL files match physical COMPILE010; production-packets.csl is retained
historical source and is not the active packets.csl module. A first-error latch
records receive phase before failure, protocol counters, leases, the complete
31-word RX bank, the retained TX frame and a source snapshot only while its
lease and role-specific length are valid. Two full row0 reads must agree.
Completion markers prove the scalar copy finished, not an atomic snapshot of
words that a DMA could update. The source-frame point observations have the
same limitation. Full protocol success still requires all original 650 checks
and stable, entirely zero fault banks.

Nine host-source files match the separately qualified synthetic capture/store
and saved-array validation exercise. run_hw.py preserves the actual qualified
runtime context adapter. Site-specific source gates, artifact binding, resource
guards and admission are deliberately not supplied as a runnable submission.
These files do not authorize or automatically submit a device job. host_tests.py
uses synthetic replies and cannot qualify physical traffic. Source and evidence
hashes in the report identify exactly what ran; do not substitute a later draft.

[Report](../../../../docs/FULL136-FIRST-FAULT.md) |
[Accepted failure evidence and source map](../../../../evidence/ready-fanin/native-control/full136-first-fault.json)
