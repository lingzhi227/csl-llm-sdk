# Original Layer 3: continuous positions and device reset

The complete original Layer 3 graph executes position 0, position 1 and a reset
replay of position 0 on physical WSE-3. One initialization and one weight upload
serve all three executions. The complete captures pass the conditional operator
audit, including KV evolution and exact replay after device reset. This is a
three-execution Layer 3 result; full-model inference remains unqualified.

## Execution and retained state

| Serial | Request | Reset generation | Position | Reference input row | Valid KV slots |
|---|---|---|---|---|---|
| 1 | 1 | 1 | 0 | 0 | 1 |
| 2 | 1 | 1 | 1 | 1 | 2 |
| 3 | 1 | 2 | 0 | 0 | 1 |

The host supplies the two existing reference input rows. No neural intermediate
or KV upload occurs between phases. Reset happens after quiescence on device;
weights remain retained, the private execution serial remains monotonic, and
the new generation restarts the position at zero. The replay output and active
K/V slot exactly match serial 1. Inactive slot 1 retains its serial-2 bytes and
is masked by the valid length, rather than being cleared. All six replicas of
each canonical KV head agree.

Both initial and final parameter-retention passes verify all 195 device arrays.
The prepared input record contains 103 files: 95 weight files and eight other
descriptor, route, normalization, frequency and reference files. All three
executions have complete neural archives, all-PE lifecycle records and two
stable transport snapshots. Each archive is made durable before its storage is
reused. The final capture has 1,799 neural files totaling 195,645,300 bytes.

The exact operation contract contains 2,387 logical copies and eight launches.
The physical journal records 12,742 copies and 2,488,475,964 host bytes. The
3,626-file raw inventory totals 238,147,178 bytes, including 1,811 binary files
totaling 228,048,468 bytes, 9,807 durable rows and 32,478 journal events. All row,
aggregate, prefix and journal checks pass with no scratch or uncommitted files.

The device client exits normally and its scheduler job reaches SUCCEEDED.
Independent checks establish no remaining assignment or owned processes, and
all raw evidence is durably backed up. The guarded lifecycle takes 393.486
seconds, including initialization, extensive diagnostics and release. This is
not a token-latency measurement. The billing ledger was not queried.

## Preserved supervisor failure and separate audit

The original outer supervisor exits with code 1 after all three captures. Its
exact copy/launch comparison compares a saved JSON list with a freshly built
Python tuple for the schedule. Independent reproduction finds that this is the
only differing field; the entire contract agrees after JSON normalization.

The original failure and absence of a COMPLETE record remain unchanged. A
separately frozen, admitted offline package validates the immutable source and
receipt pins, normal device release and all checks that the supervisor had not
reached. It then runs the original numerical core. The only audit adaptations
are the normalized metadata comparison and explicit validated postcapture
provenance as input; arithmetic, thresholds and references are unchanged.
Outputs belong to the separate package. No device retry or replacement
completion record was made. The offline process exits with code 0.

## Compiled storage and ownership

The 750 by 90 application retains 67,500 PEs, including 33,750 original neural
PEs, 30,576 matrix PEs, 24 attention heads and 136 MLP owners. Inspection covers
all 1,299 actual application programs, 18,336 task entries and 15,138 named live
banks. The descriptor models, receive-base identities and scalar packet
compaction are checked against previously accepted programs.

The maximum ordinary allocation plus 4,096 reserved stack bytes is 48,288 bytes.
This candidate has a separately reviewed 48,640-byte ceiling, leaving 352 bytes
under that ceiling and 864 bytes below physical capacity. The additional spare
policy is 512 bytes. The previous 48,128-byte ceiling and its failed candidates
remain in the history; this is not a global policy change.

The private execution counter is narrowed to 16 bits for the bounded 0-to-3
schedule. Actual code confirms the narrow storage and required widening at
arithmetic and API boundaries. The static stack analysis is conditional on
unproved startup, task-context and return-ABI assumptions; it is not a measured
dynamic stack peak or a formal whole-program memory proof.

## Numerical evidence and limits

The postcapture audit checks 11,741,184 local conditional matrix rows and 91,728
predetermined exact FP32 FMA samples comprising 8,805,888 multiply slots. All
implemented operator gates pass across all three serials: reduction chains,
BF16 conversions, handoffs, normalization, all 24 QK/RoPE and attention paths,
KV prefixes/tails/replicas, MLP nonlinear operations and both residual additions.
Its numerical work takes 37.421 seconds; the full offline guard takes 48.776
seconds with a sampled peak of 285,016,064 bytes, within its unchanged limits.

| Serial | BF16 mismatches against nominal CPU / 5,120 | Maximum absolute error | Maximum relative error over nonzero reference |
|---|---|---|---|
| 1 | 346 | 0.00390625 | 0.3333333333333333 |
| 2 | 1,150 | 0.001953125 | 1.0 |
| 3 | 346 | 0.00390625 | 0.3333333333333333 |

The maximum error at zero reference values is 0.0001220703125 for each serial.
These are reported differences, not an end-to-end acceptance tolerance.
Conditional checks use observed operands; no source-propagated whole-layer
enclosure or bitwise CPU parity is established. Archived standalone
trigonometric probes do not provide an additional numeric gate.

An independent reader reconstructs another exact FMA row at every matrix PE for
each serial, all reductions and handoffs, residuals and state/KV/reset behavior.
It uses the same qualified integer FP32 engine, so it is independent selection
and reconstruction, not a second arithmetic engine.

Embedding, recurrent Layers 0–2 and their connection to Layer 3 are next. All 64
layers, full-vocabulary logits, generated tokens and checkpoint restoration
across the three sequential stages remain open. This 90-row verification layout
cannot simply be repeated 20 times within the 1,172-row fabric.

[Exact compiled device sources](../examples/native_layer3/continuous_reset) |
[Full operator audit](../evidence/native-layer3/continuous-reset/audit.json) |
[Scope and acceptance pins](../evidence/native-layer3/continuous-reset/summary.json)
