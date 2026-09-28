# Complete MLP preparation and transport

P28 compares the complete original layer0 MLP against accepted physical
`layer-mlp-hw-003`, using the same immutable four-case independent oracle.
The shape remains5120 ->17408 ->5120 with all original banks resident across
the78x146 stage. Mixer/state operations remain inactive. This component does
not establish a complete layer, sentence generation or aggregate model TPS.

## Changed dataflow

The40 initial input quantizers previously did their arithmetic only after
receiving an exclusive return-bus grant. Each now receives a prepare-only
command after the controller start timestamp. Quantizers on separate PEs can
work concurrently while retaining their private67-word result frames. A
prepare command confers no injection permission. The existing186 response
grants still serialize actual bus injection; the per-recipient FIFO delivers
preparation before the frame fetch. A sender retains its frame through local
DMA and output-queue flush before restoring the single transit receive route.
Its prepared flag must clear before warm rearming. Sender counters distinguish
received commands from actual flushed frames; the controller expects226 commands
and186 responses per epoch.

Controller native-input packet construction now uses strided16-bit vector
stores: one stream writes selectors into upper halfwords; another copies native
operands into lower halfwords. Epoch/part/scale headers retain the original bit
layout. Final output segment collection uses32-bit vector copies. Original
arithmetic, tensor ownership, native FMA order and BF16 boundaries are unchanged.

Correction to the P27 next-step hypothesis: fused activation quantization already
runs when the last of128 values arrives in `layer_fusion.put`. Its grant-time
`prepare_group` only copies an already-quantized result. P28 does not claim to
introduce that existing overlap. The new overlap concerns the40 initial groups.

## Timing and checks

The controller records its48-bit counter at start, after the last initial-input
slice send, after the last fused-activation slice send and on final output
receipt. These phases overlap native computation and are not isolated kernel
durations. Host completed-output time starts before arm and ends after all5120
outputs return through a blocking device-to-host copy. Initial boundary uploads,
startup and original-bank initialization are reported separately. The earlier
host arm/start/finish RPC interval remains recorded as a distinct field.

Comparisons retain all four cases and require exact oracle identity, bit-exact
outputs, native ingress/scales, worker/sender counters, resident-bank/table
retention, all-PE drain and normal stop. Counter ratios are not calibrated wall
speedups. P27 did not record the explicit completed-output timer, so its short
RPC interval cannot supply a wall-time denominator. No clock frequency is assumed.

All127 local tests pass, including delayed quantizer/FIFO ordering, private-frame
ownership through warm reuse, all full/tail input wire layouts and rejection of
incomplete or mismatched performance comparisons. Fresh physical compilation
admits4,179 ELF images covering11,388 PEs, with maximum48,128 bytes including the
unchanged4,096-byte stack reserve. Three PEs have zero remaining margin. These
compiler and source checks do not by themselves qualify neural execution.

## Remaining centralized communication

`layer-mlp-traffic-001.json` binds a static traffic audit to the frozen candidate
network. Each epoch still sends320 initial-input frames (11,840 words) and544
fused-activation frames (37,536 words) from one controller. There are678 grant
words and14,382 response words. Native consumers receive343,040 operand halfwords,
including the actual full and ragged K owners. Fixed cardinal multicast branches
carry traffic even when a recipient's RAMP filter rejects the selector.

These are scheduled word counts, not measured link utilization, elapsed cycles
or model TPS. The audit excludes native reduction, projection/fusion ingress and
SDK traffic. Its purpose is to identify which distribution paths should be
partitioned or connected directly after actual phase measurements are available.

## Physical comparison and release

`layer-mlp-hw-004` passes the entire unchanged four-case numerical, ingress,
counter, retention, drain and normal-stop contract on one physical WSE-3.
All20,480 final BF16 outputs remain bit-exact to the same independent oracle.

| Case | P27 controller ticks | P28 controller ticks | P27/P28 counter ratio | P28 completed-output wall time |
| --- | ---: | ---: | ---: | ---: |
| Normalized nonzero | 2,844,472 | 713,706 | 3.9855 | 2.801ms |
| Zero after nonzero | 2,358,523 | 683,492 | 3.4507 | 2.521ms |
| Changed nonzero | 2,844,301 | 713,752 | 3.9850 | 2.555ms |
| Warm replay | 2,844,443 | 713,625 | 3.9859 | 2.388ms |

The nonzero cases use about74.91% fewer controller ticks. The normalized case
splits into163,614 ticks through initial-input distribution,405,913 through the
fused distribution interval, and144,179 for remaining down/collection work.
The intervals sum to the measured713,706 ticks; overlapping native computation
prevents attributing all405,913 ticks to transport alone. The combined change
does not isolate the individual contributions of parallel preparation and vector
copies. These are four diagnostic cases, not sustained serving measurements.

SDK loading takes331.653s, original bank/table/setup initialization3.034s and the
complete diagnostic run350.365s, including retention and3.879s normal stop.
The artifact is54,481,129 bytes, SHA-256
`8fc9ba73a4c9450aa1f7ed208303ddfe9586c4c199ed4d0fcddb945e1f71bb05`.
Source and driver identities match the frozen executed snapshot. Comparison
receipts bind both baseline and candidate; actual arrays and full ELF census
remain remote with recorded hashes.

Both fresh compiler and runtime jobs succeeded and released their resources.
The final account audit reports no owned active job or assigned system; the
workstation has no active task and its heavy-job lock is free. No new workstation
heavy job was needed. Original payloads were hardlinked rather than retransferred.
The retained MLP attempt family has1,384,266,206 unique file bytes on ALCF;
provider-billed node-hours are unavailable from the accounting API.

Next work coalesces each group's native slices and overlaps next-group return
with current-group broadcast under distinct buffer ownership, then measures the
same complete original MLP. Direct partitioned producer-to-consumer paths and
residual/norm/neighboring-layer integration remain open, as does the full64-layer
correct-sentence and>=2000 aggregate generated-token/s target.
