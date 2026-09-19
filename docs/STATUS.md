# Implementation status

September 19, 2026 UTC: the complete native layer3 graph passed independent
actual placement, original storage and SRAM checks: all 33,750 PEs, 24 heads and
1,254 application ELFs, with 576 bytes minimum static headroom. Two physical
attempts verified initial parameters and report all 24 head metadata finished,
but the original layer remains incomplete. The latest first-error record rejects
a malformed MLP READY message. Both failed jobs were released. Complete original
neural-layer epochs and CSL model generations remain zero.

A separate physical 136-producer fixture now reproduces the READY corruption
without neural math. All 408 source snapshots are correct; a received payload
contains a network-header value. Strict validation failed despite a normal
runtime exit. This narrows the investigation but proves no root cause or repair.
See [reproduction report](READY-FANIN-REPRODUCTION.md). A native-control transport
now passes focused two-PE simulator checks, including actual message routing.
The [qualification report](NATIVE-CONTROL-QUALIFICATION.md) preserves the failed
suites and separates that result from physical fixture repair. The subsequent
[native-control physical comparison](NATIVE-CONTROL-PHYSICAL.md) also failed,
with ordinary value3 at the expected control-tail position.

| Component | Current accepted scope |
|---|---|
| Native control physical comparison | Full356-program compile accepted at27,872 bytes including stack; physical004 fails at a control-tail boundary,408 source records exact,13 captures saved, normal stop/release verified; no repair |
| Native control termination | Separately accepted isolated static-route cases and consecutive 31→8 packets, then actual SDK message routing with both bankA=4; complete buffers/leases/order/suffixes/counts pass; simulator only, physical136 remains open |
| Managed receive descriptor comparison | Complete 356-PE compile passes at 25,600 bytes including stack; all 13 physical captures byte-identical to combined TX on the same system; strict protocol still fails, normal exit and release verified |
| Combined-frame transmission comparison | Complete356PE compile passes at 25,600 bytes including stack; physical run still fails on the same malformed payload despite408 exact source records and normal exit; different assigned system |
| Concurrent READY physical reproduction | 442 coherent records; all 408 producer snapshots exact; malformed receive at record 21; strict check failed, normal context exit and release verified |
| Physical QK archive/alias | Four PEs, two synthetic resets, exact960archive/256output/1280source-poison each; normal stop/release |
| Selected complete native programs | 89 original positions, all24heads/24sinks, max47552 including4096stack; other programs demoted, never executable |
| Native KV/Q ownership | 45-PE SDK fixture, three operations/reset, 29,184 exact payload halfwords, 192 canaries, 525 actual API rejects, normal stop and release |
| Original layer 3 MLP | HW01: complete 5120→17408→5120 MLP, four original/changed/zero inputs, full resident weight retention and independent numerical audit on physical WSE-3 |
| Full text CPU reference | Original 64 layers, 851 text tensors, full vocabulary, four generated tokens and complete original cache restoration; CPU acceptance only |
| Full native layer 3 compiled graph | 1,254 programs, 33,750 PEs, 30,576 matrix PEs; no demotion; max 47,552 including 4,096 stack under 48,128 |
| Full native layer 3 physical prefix | Initial 195 original parameter readbacks exact; all 24 heads finished metadata; origin rejects READY word 7 = 0x00400108; no archive payload, normal stop or full-layer numerical acceptance |
| Earlier FIFO physical diagnostic | Twenty READY heads and four incomplete heads retained as historical evidence |
| Finite FIFO observation | SDK16-PE capacity 64/128 and device-gated drain fixture; physical six-source finite-prefix capture through 50 idle locations |
| Q/K/V synthetic fanout |112 roots, 24 heads, 576 packets, 192 rows and 24576 exact markers completed in the compressed SDK graph |
| Dense-stage transport | Physical 600-PE two-epoch ownership, packed BF16 bank and connected transport qualification |
| Selected original attention | WP15: selected query/KV head, two original inputs, persistent KV and independently checked handoffs/numerical stages |
| Recurrent primitives | Scoped convolution/DeltaNet/gated-normalization simulator milestones; complete original recurrent layer remains integration work |
| WP09 recurrent integration | Unaccepted; prior lifecycle/weight-readback issue remains open |
| Three sequential logical stages | Target layers 0–19, 20–43, 44–63 with host hidden/KV/DeltaNet/convolution checkpoints; complete device execution not yet qualified |
| Complete CSL text generation | Not yet accepted |

The complete graph now fits, while execution remains blocked by an observed
protocol-format error. Strict READY checks stay in place; bounded source and
transport diagnostics will distinguish buffer corruption from packet framing.
Compiled fit, completed head metadata and physical archive-fixture equivalence
do not establish original full-layer numerics or model generation.

[Complete native graph and failed runtime prefixes](NATIVE-LAYER3-GRAPH.md) ·
[Physical archive and selected fit](QK-ARCHIVE-PHYSICAL.md) ·
[Native SDK milestone](NATIVE-KV-SDK.md) · [Physical FIFO diagnostic](LAYER3-FIFO-TRACE.md) ·
[Full MLP](HW01-FULL-MLP.md) · [CPU reference](FULL-REFERENCE.md) ·
[Earlier milestones](MILESTONES.md)
