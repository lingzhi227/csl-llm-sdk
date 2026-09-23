# Original vertical Layer3: physical attention, state and reset

September 23, 2026 UTC. This milestone retains original BF16 weights, full layer
dimensions and accepted math for physical positions 0, 1, then reset and replay
of position 0. Inputs are pinned original CPU-reference Layer3 input rows; this
run has no adjacent-layer handoff and is not full-model generation.

## Placement and actual compiled resources

The 29×1160 application has 33,640 PEs: 30,576 matrix, 188 support, 639 routers
and 2,237 idle. Its 464 logical matrix stripes retain tile indices and reduction
order. A top spine connects packet endpoints. Native K/V streams use four
colors reused only in disjoint regions; all 24 heads have adjacent archive sinks.
There are 611 packet endpoints and 96 native recipient deliveries.

All 1,087 application programs, 448,951 coordinate banks and 15,502 task entries
were inspected. Maximum ordinary SRAM plus the declared 4,096-byte stack is
48,336 bytes, below Layer3's 48,640-byte ceiling by 304 bytes. Fourteen actual
descriptor-operation models match earlier qualified Layer3 models. All 138 IO
peripheral files were compared: section contents are unchanged; east-side
program-header physical-address fields reflect the narrower application.
This is not an exhaustive control-flow or dynamic-stack proof.

The capacity expression 18×30 + 6×29 = 714 motivates the vertical layout for
the 24-layer middle stage; it does not prove compiled fit for a complete stage.

## Numerical and persistent-state evidence

All 11,741,184 conditional matrix-row checks, 91,728 predetermined exact FMA
samples covering 8,805,888 multiply slots, ordered reductions, normalization,
attention/RoPE/cache, MLP and residual checks pass. All 24 heads and archives
join completion before the next position. Actual device KV is retained between
positions; runtime performs no CPU model forward or reference substitution.

Initial full KV is zero. Device reset preserves all KV bytes before replay,
as specified by the original implementation; replay overwrites the active
position. Position-0 final hidden output is bit-exact after reset. This does
not assert that inactive cache suffixes are erased or match the first execution.

| Execution | Nominal BF16 differences / 5,120 | Maximum absolute error |
| --- | ---: | ---: |
| Position 0 | 346 | 0.00390625 |
| Position 1 | 1,150 | 0.001953125 |
| Reset, position 0 | 346 | 0.00390625 |

All three final hidden vectors are bit-exact with the earlier accepted
horizontal Layer3. Conditional checks use actual captured upstream operands;
they do not provide a source-propagated whole-layer or whole-model error bound.
The controller independently verified every raw hash and journal record,
reviewed the qualified arithmetic and geometry adapters and all audit results,
and independently decoded the reference/output BF16 comparison. A distinct
second arithmetic engine was not run for this layout. Context remains capped at 8.

## Execution and durable evidence

The 16-channel upload uses 252 original weight copies in 66 batches, at most
four retained tasks and 36,495,360 retained bytes under the 64 MiB bound. Server
concurrency and speedup are not established. The worker request is 8 GiB; its
actual peak memory was not measured and no minimum-memory claim is made.

Physical execution exits normally, all owned processes are reaped and actual
device assignments are independently confirmed empty before numerical auditing.
The capture contains 13,771 copies, 2,463,237,564 host bytes, eight launches,
54,723 journal events and 11,341 raw files totaling 201,846,780 bytes. Offline
audit runs after release without the SDK. Four bounded evidence bundles preserve
12,671 unique original files totaling 424,481,588 bytes, with complete raw-file
coverage, source/destination hashes, read-only files and durable filesystem sync.

Physical guard wall time is 371.878626 seconds; the host capture window is
110.465794 seconds; offline guard wall time is 93.999516 seconds. These include
startup, diagnostics and host work and are not pure-wafer/token latency,
matched performance or verified billing measurements.

## Preserved failures

The compiler succeeds, but its original host inspector exits with a failure
because one migrated check uses Layer0's 48,128-byte ceiling. Layer3's admitted
48,640-byte policy was already present in the profile and initial SRAM check.
The original frozen code, failure and report remain unchanged. Separate bounded
inspection of the same artifact corrects only that policy application; no
recompile or policy increase occurs.

A combined resource/identity review completes its resource checks, then fails
before identity inspection when it attempts to raise a previously lowered CPU
hard limit. A separately bounded identity review passes; the failed wrapper is
retained. An initial unlaunched preparation candidate names the wrong manifest
location; corrected preparation retains all original packed words and tile order.

## Scope and next implementation

This is accepted individual-layer conditional operator and state qualification.
Actual 0→1→2→3 device handoff, complete checkpoint/restore, compiled full stages,
all 64 layers, embedding/final norm/full vocabulary, dependent generated tokens
and matched timing remain open. The target is one physical CS3 reused for
sequential 20/24/20 stages. Full-model CSL neural epochs remain zero.

[Device source](../examples/layer3_vertical/device) ·
[Host/audit source](../examples/layer3_vertical/runtime) ·
[Evidence summary](../evidence/layer3-vertical/qualification.json).
