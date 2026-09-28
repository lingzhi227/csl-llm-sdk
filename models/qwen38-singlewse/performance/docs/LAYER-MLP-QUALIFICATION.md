# Complete original MLP qualification

P27 qualifies the P26 connected spatial MLP on an actual WSE-3.
The entire layer0 stage remains78x146 PEs at logical origin63,0. The physical
component uses offset67,1 inside the762x1172 fabric, matching the proposed full
pipeline placement. The mixer banks are present and initialized with their
original weights, but mixer/state operations are inactive in this component.
This is not a full neural layer or complete model acceptance.

## Original data and independent oracle

`layer-mlp-reference-001` reads the publisher-pinned original layer0 weights.
The complete enclosing stage allocates396,953,216 bank bytes. All eight original
matrices, expanded original scales and small weight pages are initialized; unused
state/value pages start at zero. This preparation uses2 CPU cores, a2GiB memory
cap and no swap, and completed in42.416 seconds. Payloads remain remotely stored.

The independent oracle reads row-major original weights in bounded row strips.
It reproduces native32/64-column FMA rounding, local K-shard addition and the
balanced own/left/right FP32 reduction. Each complete projection is independently
bounded against a separate FP64 mathematical sum before BF16 conversion. PyTorch
supplies SiLU and independently checks actual FP8 operand codes and scales.
Neither the CSL decoder nor candidate weight-packing arithmetic is the oracle.

Four immutable cases exercise normalized nonzero input, zero after nonzero,
changed nonzero input and a warm replay of the first input. Acceptance was frozen
before device outputs: all5120 final BF16 values per case must match bit for bit;
every native input code/scale and completion counter must match; all initialized
banks and SiLU tables must remain unchanged; every PE must retire its computation
and local output before normal runtime shutdown. Host code uploads only original
banks, the immutable SiLU table, initialization descriptors and initial BF16
component inputs. It does not inject gate/up/activation/down intermediates.

## SiLU and completion

An exhaustive independent CPU proof covers all65,280 finite BF16 input values
against `torch.nn.functional.silu(BF16.float()).bfloat16()`. A3,328-entry table
uses6,656 bytes on each of33 low-payload fusion actors. Small values use exact
integer halving with ties to even; positive values at least8 retain their input
bits; sufficiently negative values return signed zero. The full proof and pinned
table hash are retained in `layer-silu-oracle-001`. Nonfinite values are excluded
and rejected. This CPU proof alone does not establish device execution.

The global component `finish` entry waits on every worker's native row loop and
each sender's actual output-queue-empty event. The controller's receipt of the
last output is insufficient by itself. Every sender records grant and flushed
frame counts, and warm rearming requires completed prior leases. The controller
captures its own48-bit start/end cycle counter; host setup, arm/start/finish and
full runtime durations are recorded separately. None is reported as model TPS.

`layer-mlp-compile-007` admits this LUT and local drain implementation over all
11,388 application PEs:4,179 ELF images, maximum47,968 bytes including4,096 bytes
reserved stack, leaving160 bytes at the most constrained image. This is bounded
workstation compilation, not physical execution. Subsequent input/scale exports
and controller timestamps require the fresh physical compiler/SRAM admission.

## Bounded physical attempt

`layer-mlp-hw-001` freezes generated routes, source identities, original bank
extents and the independent reference metadata. The physical profile admits a
64MiB compressed artifact only for this exact stage, shape, revision and placement;
other component profiles retain their prior limits. Extraction is bounded at
512MiB. Upload uses one bounded artifact message, and compiler download preserves
multi-chunk framing. Every ELF and application coordinate is audited before
runtime allocation. Compilation and runtime have separate900-second deadlines,
shared hardware-lock admission, a1GiB sampled host-RSS ceiling and correlated
cleanup. The source tests pass122 checks.

The first payload transfer reached its180-second transport deadline before any
hardware dispatch. Recovery verifies all existing prefix bytes against the frozen
remote source and appends only missing bytes, followed by full manifest checks.
The corrected transfer deadline is600 seconds per file; this does not change
neural acceptance or device runtime limits. Mac retains only sources/metadata.

## Physical result

`layer-mlp-hw-003` passed all four cases on one physical WSE-3:20,480 final
BF16 outputs match the frozen oracle bit for bit. All native ingress codes/scales,
worker/sender counters, initialized bank/table retention and actual all-PE drain
passed. The warm replay passed and the runtime stopped normally. All changed
CSL and the bounded host driver match the frozen executed source.

| Case | Controller timestamp difference |
| --- | ---: |
| Normalized nonzero | 2,844,472 |
| Zero after nonzero | 2,358,523 |
| Changed nonzero | 2,844,301 |
| Warm replay | 2,844,443 |

These are device cycle-counter differences from controller start through final
output receipt. No cycle-to-seconds calibration was performed. The recorded
1.48–1.99ms host arm/start/finish call interval is not used to infer device
frequency, completed inference latency or model TPS. Future speed comparisons
must retain an explicit completed-output timing boundary as well as device ticks.

SDK runtime preparation took323.486s; original bank/table/setup initialization
took3.122s; the full runtime including four cases, diagnostic readback, complete
retention checks and normal stop took340.388s. These startup/diagnostic durations
are separate from the spatial computation. Full-model TPS remains null.

The physical artifact contains4,179 ELF images covering11,388 PEs. Its54,461,067
compressed bytes and293,212,959 expanded bytes fit the declared bounds. Every
actual PE passes the48,128-byte application ceiling including4,096 reserved stack;
the most constrained image has zero margin. Sparse shared load metadata produces
two ELF files larger than1MiB, although their PE SRAM still fits.001 preserves
that host admission failure;002 admitted the unchanged artifact with a2MiB ELF
file-size guard and unchanged SRAM/total-artifact limits.

002 then failed before neural execution during client gRPC request serialization
and was cancelled/released. A no-cluster-job test using the actual54MB artifact
and real protobuf/gRPC loopback found that mapping the397MB bank file caused
protobuf deserialization failure under4GiB address space; the unmapped case passed
with a peak341,252KiB RSS. This is evidence of address-space pressure, not an exact
reproduction of the original serialization error.003 reads bounded physical-row
slices without mapping the complete bank, retains the same4GiB address-space and
1GiB sampled-RSS limits, and passes. No neural criterion or CSL changed.

The compiler service001 and successful runtime003 are SUCCEEDED/released; the
failed runtime002 is CANCELLED/released. The final account snapshot has no owned
active job or assigned system. Provider-billed node-hours are not exposed by the
available accounting fields. Identical banks/artifacts/accepted ELF images are
hardlinked across retries; retained unique ALCF file bytes total1,024,586,890.
All three P27 workstation tasks are released and retain697,150,188 bytes.

The full64-layer resident pipeline, neighboring-layer/state integration, correct
sentence generation and measured>=2000 aggregate generated tokens/s remain open.
See `layer-mlp-qualification-summary-001.json` for bound result/raw/source hashes.
