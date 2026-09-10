# Accepted milestones

Only accepted results are listed below. Ongoing WP09 integration and WP10 attention work are tracked in [implementation status](STATUS.md).

## WP08 — Accepted, September 10, 2026

Repaired one-PE selected-head preprocessing passed eight tokens, full width4
history, all numerical/conversion gates and reset checks. The first failed
beta-source/exp candidate is excluded; thresholds and fixtures were unchanged.
See [report](WP08-REPORT.md).

## WP07 — Accepted, September 10, 2026

Standalone BF16 direct-gain gated RMS128 and its third-PE recurrence composition
passed all declared stage/conversion, state and completion checks. Four calls
and four composed tokens fit the same SRAM/host limits. Only the observed
successful join order is device-qualified. See [report](WP07-REPORT.md).

## WP06 — Accepted, September 10, 2026

One full128×128 synthetic recurrent head passed four token updates across three
generations on two PEs. All prediction, delta, full-state and output gates passed;
state persistence, resets and framed transfer lifecycle were verified. Both PEs
fit48 KiB including4 KiB stack allowance. See [report](WP06-REPORT.md).

## WP05 — Accepted, September 10, 2026

Four full5120 ordinary RMS calls and64 signed conversion probes passed with
offset gain and actual BF16 RNE. Shared gain/output storage plus a4 KiB stack
allowance fits the48 KiB application budget. The first noncompliant SRAM
candidate was stopped and excluded. See [report](WP05-REPORT.md).

## WP04 — Accepted, September 10, 2026

Pinned source/metadata audit matched851 text tensors and established branch-specific
equations. Twenty-seven bounded CPU checks passed extracted official function bodies
against small independent references. Full runtime/model integration remains open.
See [report](WP04-REPORT.md) and [semantic contract](WP04-SEMANTICS.md).

## WP03 — Accepted, September 10, 2026

Original BF16 rows 0:128 span all 5120 input columns using device-owned persistent
FP32 accumulation across 46 tiles. Four calls passed all boundary/final, state,
nonzero-tail-padding and guard checks, including exact one-hot at global5119 and
zero-after-nonzero. Simulation completed in 205.8 seconds with normal stop/exit.
Scope is 128 outputs, not the entire 17408-row projection or model.
See [report](WP03-REPORT.md) and [evidence](../evidence/wp03.json).

## WP02 — Accepted, September 10, 2026

Two adjacent PEs split the original tile into two 56-column contractions, send
partial FP32 results over fabric, SUM on the receiving PE and normalize 128 elements
with unit gain. Four calls passed all independent stage/approximation/state gates
and exercised local-first and receive-complete-first joins. Normal SDK stop/exit
was recorded. This is an on-wafer reduced chain, not physical cluster inference.
See [report](WP02-REPORT.md) and [evidence](../evidence/wp02.json).

## WP01 — Accepted, September 10, 2026

An original 128×112 BF16 up-projection slice completed four calls in one runtime,
including changed input, exact last-column one-hot and zero-after-nonzero. All
weight/guard/counter/handle checks passed; normal SDK stop and process exit were
recorded. See [report](WP01-REPORT.md). This is a local GEMV tile, not full inference.

## WP00 — September 10, 2026

Resource admission, serial heavy-task locking and three host bit codecs; a single
WSE-3 PE copied seven BF16 raw bit patterns with native MEMCPY_16BIT on SDK 2.10.1.
H2D, device copy and D2H preserved `0000 8000 3f80 bf80 7f80 ff80 7fc1`.
SDK stop returned normally and both compiler/simulator processes exited zero.

Outer wall times were 4.083871 s for compilation and 2.058936 s for simulation.
These include container/service overhead and are not hardware performance.
The first attempt failed before compilation due to host/container mounting; the
controller approved a corrected launch with unchanged CSL/driver source.

This milestone establishes a tiny transfer control, not BF16 arithmetic, request
reset, packed-stream SDK support, model generation or physical cluster inference.
Fourteen lightweight host tests pass. See [WP00 report](WP00-REPORT.md) and
[compact evidence](../evidence/wp00.json). Future entries require scoped acceptance.
