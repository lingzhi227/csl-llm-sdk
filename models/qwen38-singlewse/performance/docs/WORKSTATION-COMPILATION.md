# Workstation compilation resources

The user requested correcting repeated undersized compile limits on 2026-09-28.
Complete resident-layer builds in `tools/stage_layer_mlp_compile.py` now default
to 16 GiB RAM and a 1,800-second compiler deadline (1,830-second service limit).
Before dispatch, the stager requires `MemAvailable >= MemoryMax + 8 GiB` to
leave host headroom. It rejects insufficient capacity rather than silently
shrinking the compiler's limit. Smaller selected-component probes have separate
stagers; these defaults concern complete layer builds.

Keep the shared heavy-job lock, no swap, CPU quota and immutable attempt records.
An explicit resource override should use measured working-set requirements, not
restart the old 2/4/6 GiB escalation sequence. Dispatch receipts record the actual
memory/time budget and pre-dispatch memory observation. A timeout without ELF
results is a workstation compilation failure, not a PE SRAM rejection.

The already-running compile026 had its service MemoryMax raised from 6 to 16 GiB
without restarting it. Its frozen source and original 900-second Python deadline
were unchanged. The remote `resource-adjustment-user-001.json` records that live
change. New builds receive the longer default; no duplicate job was submitted.

This compilation generates the 78x146 layer0 candidate's specialized PE programs
and routes, then checks actual linked sections plus the declared 4 KiB stack
against each PE's existing 48,128-byte application ceiling. It does not execute
neural arithmetic, prove route liveness, measure inference speed, or qualify the
complete 64-layer model. User host RAM and PE SRAM are separate budgets.

Validation: `evidence/workstation-compile-policy-fix-001.json` checks the host
headroom boundary, rejection before service dispatch, generated defaults,
unchanged CPU/no-swap limits and recorded budgets using mocked SSH/services.
No physical job was submitted for this configuration fix.
