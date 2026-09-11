# Resident spatial SDK002 result

The eight-PE resident MLP driver completed all three inputs and stopped normally.
The overall guard remains **failed** because its final check rejected the new
identities of the three runtime-generated auxiliary ELFs. Independent numerical
and geometry-specific auxiliary reviews subsequently accepted this exact fragment. No result or frozen admission has been changed
to erase this failure, and no repeat execution is proposed.

SDK002 reused SDK001's 36 immutable compiled files, 777441 bytes, without invoking
the compiler. SDK001 and its filename-gate failure remain intact. Accepted source
preparation was reused; no original model reads, downloads or HDD access occurred.

## Completed execution

The actual SDK ELF metadata reader confirmed 17 segment rectangles per program,
all describing the correct singleton position. Ranks 0 through 7 cover physical
coordinates `(4..7,1..2)` in the 11 by 4 fabric. This gate ran before simulator
construction; it does not infer coordinates from filenames.

| Ranks / role | Ordinary SRAM end plus 4096-byte stack allowance |
| --- | --- |
| 0 / gate shard 0 | 39744 B |
| 1 / gate shard 1 | 38800 B |
| 2 / up shard 0 | 39744 B |
| 3 / up shard 1 | 38800 B |
| 4 / ingress and output | 13184 B |
| 5 / nonlinear consumer | 17264 B |
| 6 / down shard 0 | 31584 B |
| 7 / down shard 1 | 30480 B |

Every application ELF passes the strict 48128-byte ceiling. This is a static
section and declared stack check, not a measured dynamic stack peak.

The driver completed 125 operations: 121 copies and four launches. The copies
comprise nine H2D and 112 D2H calls, 581280 host bytes and 310232 native bytes.
Six initial weight uploads were followed by three generations without weight
reload. All six final complete weight readbacks match the initial original BF16
weights and endpoint guards. This establishes the observed initial/final identity,
not continuous bitwise observation at every intermediate instant.

For each generation, the driver reports 768 source/conditional rows and 640 exact
FP32-to-BF16 cast checks, with exact device handoffs, epochs, release and packet
counts. The immutable accepted source intervals were not modified. Dense,
changed and zero inputs each passed the frozen checks before input reuse. Host
neural feedback dependency is zero; intermediate diagnostic D2H copies remain
part of this qualification and its budget.

The initial archive and three generation archives contain 106 arrays. A separate
workstation-only final-weight archive contains six arrays. The post-exit audit
reconciled all 112 arrays and 316760 raw bytes with their completed transactions.
The driver recorded normal stop after all checks.

## Preserved auxiliary failure

| Runtime-generated file | Bytes | SHA-256 |
| --- | ---: | --- |
| MEMCPY_XY_ROUTES.elf | 824 | `f4a0045a5e14374c9e658a57b674ee6dba35834620f8ec10b4eab755dc915fce` |
| coord.elf | 1400 | `cd4517a60793f3ff253d88145f92a55622b16b2668db94f7057599ce14d53ea3` |
| default.elf | 52720 | `2e5d06e7c0c7436e94221f115d84e15b752cbea5afb6452276d90332a03dca6e` |

The set contains exactly the three expected names and totals 54944 bytes, below
the unchanged 54952-byte bound. Their hashes differ from the old application
identities, so the frozen gate correctly preserved a failure for review. All 36
original compiled files in SDK002 and the SDK001 parent are unchanged. Independent review found unchanged default PT_LOAD payloads/virtual addresses,
exact 11-column and 4-row coordinate values, and memcpy configuration-only
segments. The exact new identities were qualified for this geometry; arbitrary
future layouts do not inherit those hashes.

## Resource and evidence record

The single simulator unit ran for 167.246074122 seconds within the 420-second hard
limit and 400-second plan. Its actual controls were 512 MiB RAM, zero swap,
128 tasks and CPU0. Across 297 affinity polls, 6207 live task samples used CPU0.
Maximum observed or reported cgroup memory was 192991232 bytes, approximately
184.05 MiB; sampling may miss the true final peak. The owned unit is absent and
inactive with PID0, its cgroup is empty, and the global heavy queue was empty at
the post-exit audit. There is no active executor job.

The final guard exit was 1. There was one simulation, no new compilation and no
retry. The candidate contained 125 files and 2118700 bytes before the audit.
Fifteen nonparameter files totaling 354620 bytes, including the four non-weight
archives, were copied to Mac and hash checked. All learned-weight archives and
ELF binaries remain on workstation.

The 59002-byte post-exit audit has SHA-256
`c4c5c9e4ff2a71ba947be90f1dd1426f1944768a6cb0650b5ca6faa10b9711bd`.
Public numerical observations, independent acceptance and source hashes are in
[the evidence record](../evidence/wp16-resident.json).

This run exercises resident shards, device reduction/consumer ownership and
reuse across inputs in a reduced spatial graph. It does not establish full WP16,
full-model inference, physical three-system communication or hardware latency.
Simulator wall time includes instrumented readbacks and final weight retention
checks; local device cycle counters are not a measured end-to-end token latency.

## Independent acceptance

The controller accepted the exact eight-PE, three-generation fragment after a
0.217-second saved-output audit (41160 KiB peak RSS): 2304 integer-checked
conditional dot rows, 196608 products, 1152 pair rows, 1920 casts and 384 Decimal
nonlinear rows, all handoff/state/timing checks, all original initial/final weight
identities and cleanup. No SDK or original-source execution was repeated.
The original guard exit1 remains part of the accepted evidence history. Full
MLP/model and physical three-wafer inference remain unaccepted.
