# Equal-work native tile candidates

This is a separate four-PE probe. It does not change the
published native kernels or the P17 complete-matrix candidate. Its purpose is to
measure a manual choice at the partitioning level before changing global owner
identities or communication routes.

| Rows by K columns | Weight elements | Scalar input callbacks | Input packet words | Partial result words, including count |
| --- | ---: | ---: | ---: | ---: |
| 2 by 128 | 256 | 128 | 65 | 3 |
| 4 by 64 | 256 | 64 | 33 | 5 |
| 8 by 32 | 256 | 32 | 17 | 9 |
| 16 by 16 | 256 | 16 | 9 | 17 |

All candidates retain 110 FP8 and 13 BF16 slots: 35,256 original payload bytes
per PE. The selected FP8 slot 0 and BF16 slot 6 are freshly packed from the
pinned original layer-0 gate and in-projection matrices. The selected tile starts
at original row/column zero, stays within one 128 by 128 FP8 scale block, and
uses its original scale. Remaining slots preserve representative original
2 by 128 background tiles. This mixed-format component is explicitly not a new
complete-model bank format or a reuse of frozen P15 tile identities.

Each shape executes 256 multiply-accumulates per invocation. Their original
matrix slices and output dimensions differ; this measures local cost at equal
work, not matched complete-matrix latency. Each measurement repeats 32 times
and retains loop, branch, counter and timestamp overhead. FP8 mode 0 excludes
one-time weight decoding; mode 1 includes the same exact decoder in every
invocation. BF16 expands original high halves and accumulates in FP32. No
communication, tree reduction or model-token rate is included.

Eight frozen input cases cover both types with mixed, signed-zero, tiny-normal
and boundary-normal values. Independently read row-major weights feed an ordered
FP32 reference and FP64 product bounds. Every active output, zero tail, original
input packet, repetition counter and complete resident bank must pass, followed
by normal shutdown. Actual compiled SRAM admission retains the 4 KiB stack
allowance and 48,128-byte ceiling. Simulator preparation, compilation and run
each have 240-second limits under the shared two-core, 2 GiB, no-swap service.

A faster local shape is only a candidate for subsequent complete-matrix tests.
For the same 48 by 5,120 matrix, changing from 2 by 128 to 8 by 32 keeps 960
workers but increases K participants from 40 to 160, reduces output roots from
24 to 6 and increases each reduction packet from 3 to 9 words. It therefore
changes routes, colors, summation order and intermediate storage. Original
matrix coverage, independent numerical bounds, communication timing and actual
SRAM must all be qualified again before selecting it for production lowering.

## Simulator qualification

`native-shapes-sim-001` passes all eight input cases and twelve timing modes,
including exact ordered outputs, independent FP64 bounds, untouched output
tails, input/counter checks, complete original bank readback and normal stop.
The shared workstation service and lock are released. Physical results are
recorded below. These simulator observations are not measured WSE-3 rates.

The first sweep shows that FP8 does not monotonically improve with larger row
vectors: the scalar output-scaling loop grows with the row count. Attempt 002
therefore places eight PEs in one executable: four original scalar-scaling
shapes and four corresponding vector-scaling shapes. Matched pairs use
bit-identical original banks, inputs and references. Three vector multiplies
retain the original FP32 operation order and all three rounding boundaries;
scales are not reassociated or combined. All eight cases and twelve modes pass
exact outputs, matched-pair identity, full retention and normal stop.

| Shape | FP8 scalar scaling | FP8 vector scaling | Vector scaling plus decode | BF16 |
| --- | ---: | ---: | ---: | ---: |
| 2 by 128 | 527.9375 | 517.9375 | 758.9375 | 980.9375 |
| 4 by 64 | 383.9375 | 331.9375 | 572.9375 | 662.9375 |
| 8 by 32 | 448.9375 | 312.9375 | 553.9375 | 536.9375 |
| 16 by 16 | 641.9375 | 320.9375 | 561.9375 | 458.9375 |

Values are mean simulator cycles per local invocation over 32 repetitions;
they retain the loop and counter overhead. BF16 is unchanged in the paired
program and has identical measured intervals on both copies. The vector-scaled
FP8 8 by 32 shape is fastest in this sweep, while BF16 favors 16 by 16 locally.
These differing optima do not establish a complete-model partitioning choice.
Attempt 002 finishes in 62.129 seconds including preparation/compilation, and
the maximum actual SRAM plus declared stack is 46,560 bytes. The workstation
service and shared lock are released. A paired physical test is staged; the
earlier four-PE hardware staging is retained without dispatch to avoid a
redundant physical measurement.

## Physical qualification

`native-shapes-hw-002` passes on one WSE-3 with eight PEs. Eight frozen inputs
produce twelve timing modes and 3,072 counted native invocations. All 720 active
output values, zero tails, 6,240 packet words, counters, matched scalar/vector
outputs and 282,048 original resident-bank bytes pass. Both jobs succeed and
release normally. Actual maximum SRAM plus stack is 46,560 bytes, leaving
1,568 bytes; this does not include the future communication program.

| Shape | FP8 scalar scaling | FP8 vector scaling | Vector scaling plus decode | BF16, scalar-template PE |
| --- | ---: | ---: | ---: | ---: |
| 2 by 128 | 537.03125 | 523.03125 | 776.40625 | 981.0625 |
| 4 by 64 | 399.03125 | 337.03125 | 590.40625 | 664.0625 |
| 8 by 32 | 472.03125 | 318.03125 | 571.3125 | 505.0625 |
| 16 by 16 | 683.03125 | 327.15625 | 580.28125 | 444.0625 |

These are same-PE physical cycle intervals divided by 32 for the mixed case,
with all loop/counter overhead retained. The same 16 by 16 FP8 shape improves
2.08778x with vector output scaling, or 1.61172x when each call includes weight
decoding. Those are matched same-shape comparisons with exact output identity.
Across different shapes, 8 by 32 is fastest for FP8 and 16 by 16 for BF16 in
this local sweep. Different shapes cover different original matrix slices;
these comparisons do not establish equal complete-matrix latency. The BF16
paired copies differ by at most one cycle per invocation despite unchanged
arithmetic, so no FP8-scaling benefit is attributed to BF16.

The runtime lifecycle is 271.575 seconds, including approximately 198 seconds
queueing. The scheduler records Running from 20:27:43 to 20:28:32 UTC, on
2026-09-27. These job timestamps are not provider-billed node hours or kernel
latency. The fresh account/system audit confirms no owned allocation after
completion. See `native-shapes-summary-001.json` and the frozen job timeline.
