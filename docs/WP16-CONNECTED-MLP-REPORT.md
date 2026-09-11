# WP16 connected generation reuse — accepted partial MLP

Accepted September 11, 2026 (UTC). Four CSL processing elements execute an
original-weight partial MLP for dense, changed and zero inputs in one SDK runtime.
Each generation resets numerical state, computes gate/up → SiLU → product → down,
retains the observed results until explicit release, and then admits the next
generation. Intermediate numerical operands travel on device. The host supplies
weights, original inputs and control descriptors, and records observations.

## Scope and identity

The pinned model is `Qwen/Qwen3.8-27B`, revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, layer 3 MLP. Gate/up use original
rows `[0,128)` and input columns `[0,112)`. Down uses original output rows
`[0,128)` and intermediate channels `[0,128)`, accumulated as two 64-column
contributions. Gate/up weight tiles remain resident across all three generations.
Generation/input/contraction identities are `(1,0,1)`, `(2,1,2)` and `(3,3,3)`.

Candidate `wp16-connected-sdk-003` has 89 frozen files / 1,581,160 bytes,
source manifest `b7088423521080ec338c86be358e9c5c008a0e473ec4b7757165faabb7a26703`
and sole admission `d59e4c62424d63992b4833381b01a6a9f8659a1e0158af28955c242bf385a826`.
See the [result and identity record](../evidence/wp16-connected.json),
[source mapping](../evidence/wp16-connected-source-map.json) and
[published implementation](../examples/wp16/connected/README.md).

This profile differs from the previously accepted one-input diagnostic with
5,120 projection columns. Their combined coverage does not establish an executed
full-width reusable program. Full 5,120-column connected reuse, all 17,408 MLP
channels, complete-model inference and physical WSE or cluster execution remain
unqualified. Overall WP16 and the separate WP09 cleanup issue remain open.

## Device execution and ownership

PE0 computes gate, PE1 computes up, PE2 computes whole SiLU and product, and PE3
owns the persistent down accumulator. Projection, SiLU, product and final down
outputs cross explicit FP32-to-BF16 round-to-nearest-even boundaries. The source
interval policy, FTZ enclosure, nonlinear allowances and kernels retain their
previously accepted hashes; no tolerance was widened for this run.

Versioned data/ACK frames bind generation, original input, contraction and selected
ranges. Active send/receive state persists until its callback. Completion requires
the consumer acknowledgment, and retained observations must be durably recorded
before the host overwrites a tile or releases a generation. All four PEs complete
the release barrier before the next generation's input is copied. The current
range descriptors are fixed to this selected scope; a general channel/output
block scheduler is future work.

There are 696 operations: 647 copies and 49 launches, including 99 H2D and 548 D2H
copies. Host buffers account for 978,168 transferred bytes; native transfers account
for 539,652 bytes. The 9 handoffs contain 1,413 fabric words and 1,884 word-hops.
Every generation has 56 reset reads, 31 retention reads and 16 release reads.

## Independent numerical and lifecycle acceptance

The controller's independent auditor checked all frozen files, 20 compiled files
and 3 expected runtime auxiliaries. Without importing candidate numerical helpers,
SDK or Torch, it checked 6,144 source interval values, 2,688 original-weight
conditional contractions, 768 actual-operand scalar products, 1,920 exact casts,
1,152 transferred values, 36 wire frames, 42 numerical states and 15 timestamp
pairs. It confirmed 12 complete observation groups before overwrite, 81 retained
numerical/protocol buffers and 12 released handoffs. The driver's retention count
of 93 includes those 12 lifecycle observations.

The changed generation has 384 distinct gate/up/down rows relative to the preceding
generation; the final zero generation has 384 exact numerical-zero rows. Raw
signed zero bits remain present in the observations and cast checks. The separately
accepted original source preparation contains 11 interval arrays, verifies 512
dense anchor endpoints and uses the same original dense/changed/zero fixtures.
These are scoped interval and stage checks, not a claim of full-model bitwise parity.

All nine actual joins were command-before-receipt. Opposite arrival orders are
covered by host fixtures only. Complete resident projection weight payloads were
read before generation 1 and after generation 3. Per-generation guards and source
checks do not prove continuous bitwise residency or exclude every transient
within-bound corruption.

## Resources and shutdown

| Stage | Actual limit | Wall time | Maximum sampled memory |
|---|---|---:|---:|
| Compile | 1 GiB, 300 s hard | 6.770251 s | 449,986,560 B |
| Simulate | 512 MiB, 400 s planned / 420 s hard | 221.818410 s | 187,584,512 B |

Both stages used one worker, CPU 0 affinity, zero swap and at most 128 processes.
CPU affinity is not exclusive CPU ownership or a bandwidth quota. Memory figures
are sampling lower bounds, not measured true peaks. Compile had 12 affinity polls
and 204 live-task checks; simulation had 397 polls and 8,307 live-task checks.
All memory/OOM/process-limit events were zero.

Compile unit was `qwen38-job-9844b724f87741169144595d77285997.service`; simulation
unit was `qwen38-job-07a4d88e71ad40f8a4c5cdadbd7fcecd.service`. Normal SDK stop and
guard exit 0 were observed. Both owned units were independently absent/inactive,
with PID 0 and empty cgroups, and the global queue was quiet after exit.

The 20 base compiled files total 409,302 B; the 3 expected runtime auxiliaries total
54,952 B. Actual ELF ordinary section ends plus the unchanged 4,096-byte stack
allowance are 49,008 / 49,056 / 23,200 / 38,208 B against 49,152 B per PE. PE0/1
have only 144/96 B spare. This allowance is not a dynamic stack peak measurement;
new scheduler state requires a new SRAM design and actual compiled admission.

The earlier partial diagnostic's 285.327 s and this run's 221.818 s have different
shapes, generation counts and observation workloads. They are not a speedup
comparison. Blocking copy intervals can include queued computation and cannot be
read as isolated transfer bandwidth.

## Preserved failures and host regressions

Connected001 compiled but failed SRAM admission before simulation: PE0/1 exceeded
the application limit by 368/416 B. Connected002 reduced unused timing storage;
actual compiled allocation saved 512 B on each of those PEs while retaining the
4 KiB stack allowance. Simulation then failed at operation 73 because a generic
guard-snapshot predicate flushed a reset archive early and tried to create it again.
No projection or completed generation ran in that failed simulation.

Connected003 changes only that host flush predicate and candidate identity from
002. Device code, numerical checks, operations and limits remain unchanged. Its
host fixture runs the real driver loop, checks and evidence persistence with a
synthetic SDK transport. It verifies all 696 operations, durable writes before
overwrite/release, failure stop behavior, and reproduces the exact old driver bug.
The [old driver](../qualification/driver002-reproduction.py) is published solely
as a failed host regression reference. Synthetic tests are not additional device
executions. On the failed path the archive counter includes attempted writes;
the accepted run's 39 successful archives match the filesystem exactly.

## Public evidence and reproduction

The exact 1,455-record journal is 793,346 B, SHA-256
`9d3a1b47b77c4795e67cd0fa169d7d6fdf582048a308135ed998996ea9ff43d9`.
The original private 39 archives contain 548 raw arrays / 660,392 B. Thirteen
learned-weight readbacks / 524,392 B are omitted by exact operation classification.
The public 37 observation JSON files preserve the remaining 535 arrays / 136,000 B,
with dtype, shape, operation sequence and original raw-byte hashes. Eleven source
arrays / 67,584 B and the 672-byte hidden fixture are also represented losslessly.
JSON numbers round-trip to the original dtype bytes, including signed zeros.

Weight values, mixed raw archives, vendor binaries and private machine paths are
excluded. Weight identities, tensor ranges, hashes and copy metadata remain.
Forty mapped source files are exact accepted bytes; two host configuration files
are explicitly unexecuted portable templates; one is the exact failed002 driver
reference. The source map identifies every category and both accepted/published
hashes. Portable templates require local configuration and a separately frozen,
admitted candidate before a new run. Publishing them does not qualify a new run.

Run the bounded, synthetic host regressions from the public repository as described
in the [example guide](../examples/wp16/connected/README.md). Original-weight SDK
reproduction requires separately acquired matching inputs, a configured SDK
installation, source/admission records and the guarded stage limits. No original
source computation or SDK run was repeated for publication.
