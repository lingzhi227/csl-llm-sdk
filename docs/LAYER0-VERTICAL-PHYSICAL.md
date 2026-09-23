# Original vertical Layer0: physical operator and state qualification

September 23, 2026 UTC. The complete original-weight Layer0 executes two original
prompt positions, 0 and 1, then a device reset and exact position-0 replay on one
physical WSE-3. This uses the accepted [top-spine device program](LAYER0-VERTICAL-TOP.md),
unchanged original BF16 weights, 31,548 matrix PEs and all convolution, DeltaNet,
normalization and MLP operators. [Runtime/audit source](../examples/layer0_vertical/physical_runtime)
and a [sanitized evidence summary](../evidence/layer0-vertical/physical-qualification.json)
identify the exact accepted scope.

## Numerical and state evidence

The post-release audit passes 12,114,432 conditional matrix-row checks and
94,644 predetermined exact per-PE FMA samples (9,085,824 FMA slots). All 48 heads
in all three serials pass the operator gates; full 128-wide recurrent prediction,
delta, state update and output replay is exact. All original neural handoffs,
complete state snapshots, two full stable transport fences per serial, parameter
retention, zero-reset checks and all 608 neural-array reset comparisons pass.
Conditional checks use actual device operands. Sampled exact matrix FMAs do not
mean every matrix FMA was individually reconstructed.

| Serial | Position | Nominal BF16 differences / 5,120 | Maximum absolute error |
|---|---:|---:|---:|
| 1 | 0 | 0 | 0 |
| 2 | 1 | 438 | 0.0009765625 |
| 3, after reset | 0 | 0 | 0 |

The independent original nominal output comparison is reported separately from
conditional acceptance. No source-propagated whole-layer enclosure, bitwise
parity at position 1, full-model token acceptance or precision reduction is claimed.
The recurrent beta/decay evidence combines validated head values with complete
actual state replay; private shard scalars were not exported or claimed read back.

## Physical ownership and evidence

One original-weight upload uses 190 copies in 51 bounded batches, at most four
retained asynchronous tasks and 36,642,816 retained bytes. All tasks complete
before initialization. The compiled artifact uses 16 DMA channels; this alone
does not prove concurrent server execution or a speedup.

The complete capture contains 6,911 host copies, 2,512,897,044 host-slot bytes,
eight launches and 27,468 journal events. All 5,312 raw files, 179,824,400 bytes,
are independently hashed. Physical execution exits normally, the job reaches
SUCCEEDED, all owned processes exit, and actual system assignments are empty.
The post-release CPU audit also exits normally without importing the SDK or
running a substitute model forward. Complete source, capture, journal, audit,
binding and prepared-manifest evidence is durably preserved in private storage.

The physical guard spans 261.568538 seconds including startup and cleanup; the
host capture window is 68.524565 seconds. The offline audit guard spans
96.218860 seconds. These are diagnostic host windows, not pure wafer latency,
per-token inference latency, matched end-to-end performance or billing evidence.

## Preserved failed attempts

Runtime001 reaches the fixed 240-second startup deadline before model upload,
launch or neural capture. Runtime002 introduces separately bounded pre-start
and device-start windows; its start RPC returns HTTP502 after about 116.27
seconds, before the device-start deadline, again without neural capture. Both
original failures and their verified cancellation/release are retained.

Runtime003 changes only the public remote-worker memory request and limit from
4 GiB to 8 GiB; all Python, model, artifact, IO, math and host limits remain
unchanged from Runtime002. The 8 GiB attempt completes. This comparison does not
prove an earlier OOM or a minimum memory requirement. Client peak RSS was
346,419,200 bytes; the remote worker's actual peak was not measured.

An initial evidence transfer ended near its 150-second bound and left a partial
destination. A separate corrected transfer retained that failed attempt,
verified and reused complete readonly files, and made the complete archive
durable with final filesystem synchronization. All final hashes were rechecked;
the failed partial tree remains preserved.

## Remaining implementation

Next are original Layer3 vertical qualification and actual 0→1→2→3 hidden flow,
then control-state restoration and compiled capacity for 20/24/20 sequential
stages. The target remains one physical CS3 reused for all 64 original layers,
embedding, final norm, the full 248,320-token vocabulary and dependent generated
tokens. The accepted full-model CSL epoch count remains zero.
