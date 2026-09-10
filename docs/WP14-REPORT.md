# WP14 execution report

The original-hidden CPU reference and first five-PE device integration were
independently accepted on September10,2026. WP09 remains unresolved. No full attention,
whole layer/model or physical system claim follows from this work.

## CPU reference001

Four unchanged WP13 hidden vectors use positions1,2,3,0. The pinned official
Q/rawgate projection/view/chunk prefix, K projection, trained ordinary RMS256 and
partial RoPE64 statements execute on CPU. Original Q/K norm bytes are reused.
The independently frozen WP13 projection BF16 uncertainty is carried through the
new trained-weight interval policy; observed SDK values never shrink its bounds.

All projection, FP32 normalization stages, BF16 product/cast boundaries and final
outputs pass the source enclosures. All2048 final BF16 values match the source
nominal in these CPU fixtures. This parity is observational, not a universal
bitwise guarantee. Onehot projection uncertainty stays zero, while subsequent
normalization has its own arithmetic uncertainty. Structural zero stays exact.

| Case | Q/K maximum full output width | Q/K single-value elements | Rejected all-zero elements per head |
|---|---|---|---|
| Dense | 0.15625 /0.140625 | 0 /0 | 252 /252 |
| Changed dense | 0.109375 /0.140625 | 0 /0 | 252 /254 |
| Last-column onehot | 0.00390625 /0.00390625 | 252 /253 | 256 /256 |
| Zero after nonzero | 0 /0 | 256 /256 | 0 /0 |

Dense cases reject251..253 cyclically misplaced elements per head. Skipping
rotation rejects12..15 of the first64 elements at these positions; it does not
prove sensitivity to every high-frequency rotary change. Near-zero intervals
contain up to30995 distinct finite BF16 values. Controller analysis puts the
full-width L2 norm at approximately4.8..5.7% of output L2 for dense cases. These
bounds remain useful only alongside projection checks, conditional actual-stage
checks and exact casts, transport, signed-zero and tail rules.

Execution used the existing qualified CPU environment: one thread, actual2GiB
MemoryMax, MemorySwapMax0 and60second deadline. It finished normally in
2.121459157seconds at208527360bytes peak memory. All15 source files and6 frozen
inputs, including the existing raw weight files, remained unchanged. The exact
owned unit was independently confirmed absent/inactive, MainPID0 and empty
control group. No new model payload, environment or GPU was used.

Source manifest SHA256:
`4d1ac7e063a625910d1c2807ee938aee36c241d62a96b8c0be4177fc94732435`.
Observations archive437596bytes, SHA256:
`01db8e1548d2d468b36562d346018df2270b9cb5ddf8aa9c35614dbfac5eb548`.
The controller independently audited12200 interval inclusions,8216 observed stage
values,3584 casts,1536 tails and all2048 final values using a separate high-precision
construction. No second CPU candidate is needed.

## First connected SDK result

The only dispatched candidate, `wp14-001-reviewed`, completed normally. It connects
the original dense5120 hidden vector and trained weights through all four Q/K
projection producers, four exact device handoffs, trained RMS256 and partial
RoPE64 at position1. All512 projection BF16 values and512 final consumer BF16
values match their official CPU references in this fixture. The largest independent
projection FP32 error is2.4665496312081814e-9, within its predeclared source bound.
Source intervals remain conservative and separate from actual-operand stage checks.

All704 copy pairs and59 launch pairs match the frozen763-operation sequence.
The consumer committed all512 input words exactly. Its conditional stages, casts,
tail rules, original parameter/control identities and source output
gates passed. After consumer completion, all four producer FP32/BF16 outputs
including guards remained bit-identical to their finalize snapshots, and all four
complete resident-weight arrays matched the original final tile and poison.
All producer release states, stable handles and normal runtime stop passed.
No rotary product or sum in this device case was zero; the conditional validator's
zero-sign predicates therefore add no exercised zero-sign coverage here.

| Resource | Observed |
|---|---|
| Compile wall time | 6.257071306 seconds |
| Compile peak memory | 498991104 bytes |
| Simulation wall time | 256.007598589 seconds |
| Simulation peak memory | 229806080 bytes |
| Producer ordinary ELF end +4096 stack allowance | 43552,43536,43552,43552 bytes |
| Consumer ordinary ELF end +4096 stack allowance | 26976 bytes |
| Runtime enforcement | 20GiB MemoryMax, Swap0, one heavy job,300second deadlines |

All42 frozen source files and23 compiled files remained unchanged. Both exact
owned units were confirmed absent/inactive with MainPID0 and empty control groups;
the heavy queue is quiet. The guard's latest resource samples retained about31GB
available RAM,93GB free SSD and325MB active cache. No new model payload or CPU
reference execution was used for this SDK candidate.

Source manifest SHA256:
`195294a045b1a321a52201abb5297983db85636c8b49e82b884e5f59856e3726`.
Device observations SHA256:
`8179b59ce45e858deebf52393340fe4010a89dc505d5b659a4ff9636dffa3070`.
The earlier `wp14-001-ready` source draft remains retained and was never run.
There was no retry or second SDK candidate. This is one selected Q/K call in a
five-PE simulator; it does not qualify V/gate attention, cache or a complete layer.

The [accepted evidence](../evidence/wp14.json) includes typed numerical arrays,
the full identified transfer journal and original archive hashes. Complete
resident-weight and normalization-parameter payloads are omitted with their shape,
dtype and hash retained; reproduce them from the pinned selected-weight download.

## Reviewed device design and estimates

See [protocol](WP14-PROTOCOL.md). Producer ranks0..3 map to global slabs0,1,4,5,
retain all5120 serial contraction columns, and send exact Q256/K256 operands to
consumer rank4. Four distinct data routes and four ACK routes serialize framed
handoffs. The consumer requires all four committed slabs before normalization.
The driver reads all four full resident-weight tiles after consumer completion,
and rereads each guarded producer FP32/BF16 output there to compare every word
with its finalize snapshot before release. Full weight reads are moved, not
duplicated. It also preserves all consumer stages. Host protocol tests cover completion order, atomic invalid-frame rejection,
stale/duplicate/missing frames and ACK-before-commit; they are not device recovery
tests. Three such host tests passed.

The fixed sequence has704 physical copies,59 launches,10940872 host bytes,
5530824 native memcpy bytes, and2336 fabric frame bytes. Maximum physical host
buffer57352bytes. It adds92 copy calls and10 launches to the measured equal-FMA
four-PE baseline. Declared arrays plus explicit kernel/timestamp storage are
31860bytes per producer and12150bytes for the consumer, before code, compiler
state, padding, memcpy allocation and4096byte stack. The five actual ELF
ordinary-end-plus-stack gates passed as recorded above.

The265second estimate is233.914793seconds measured projection baseline,6seconds
consumer allowance and25.085207seconds for the fifth simulated PE, protocol,
additional observations and runtime overhead. This includes the eight added
retained-output reads and the moved four complete weight reads. The last allowance
was unmeasured before launch; moving a read can change when queued work is paid
for even though its byte count stays the same. The completed256.01second run is
within that265second estimate and300second hard ceiling. Individual copy/launch
wall times include waiting for queued work and do not isolate kernel compute.
Both compile and simulation had300second hard ceilings. The controller admitted
one complete chain after reviewing this risk; no separate communication diagnostic
was run. Further candidates require a new dispatch.

The runtime journal fsyncs identified entry/exit records for each physical copy
and launch. Frozen source, selected weight and compiled identities are checked
before/after execution. Runtime stop runs in a finally block and preserves the
primary exception if stop also fails. The bounded supervisor owns a unique unit,
enforces one heavy job,20GiB RAM/Swap0 and resource reserves, and stops its unit.
Postflight must verify that exact unit has no PID/control group; a failed unit
keeps its failure evidence before any explicitly scoped reset of failed state.
