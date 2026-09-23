# Implementation status

The latest accepted milestone is [original vertical Layer3](LAYER3-VERTICAL.md):
29×1160 physical positions 0/1/reset/0, original weights, all conditional operator,
KV/state/transport/archive gates, and exact reset replay. All three final hidden
vectors are bit-exact with accepted horizontal Layer3. Nominal differences remain
346/1,150/346 of 5,120; context8 and finite positions remain explicit. The original
compile inspector failure is preserved and corrected inspection used the same
artifact without recompilation. All 11,341 raw files are saved and hashed.

Updated September 23, 2026 UTC.

The complete original [vertical Layer0](LAYER0-VERTICAL-PHYSICAL.md) now has
accepted physical positions 0/1/reset/0, all conditional operator gates, exact
recurrent replay and complete state/transport/retention/reset checks. Nominal
BF16 differences are 0/438/0 of 5,120, with position-1 maximum absolute error
0.0009765625. All raw evidence is preserved, owners reaped and device release
independently verified. The 4 GiB startup timeout and HTTP502 attempts remain
preserved; the successful attempt uses an 8 GiB worker and 16 channels without
proving an OOM cause or minimum memory requirement. No full-model, propagated
whole-layer error or matched token-performance claim is made.

Current work is actual wafer-resident 0→1→2→3 hidden flow, parameterized layer
lowering and full checkpoint/control restoration for one physical CS3 reused across
20/24/20 stages. All 64 layers and full-vocabulary generation remain open.

The [finite dense-transport fixture](DENSE-TRANSPORT-FINITE136.md) is accepted
on physical WSE-3: 136 PEs, positions 0/1/reset/0, complete rows, retained
parameters and stable ownership boundaries. Independent reconstruction from
actual operands checks 1,179,648 ordered FMAs with zero mismatches. All owners
are reaped and both evidence backups independently verified. This synthetic
component does not qualify a complete layer or full stage. Held-ACK and capture
wall times are diagnostic measurements, not token performance. Earlier compiler,
startup, log-export and host diagnostic failures remain preserved.

The [finite Layer0 arithmetic fixture](LAYER0-ARITHMETIC-FINITE9.md) is accepted
in the SDK simulator: all 35,181 exp inputs, six synthetic computations, full 128
ordered recurrence, persistent state, parameter retention, zero resets and exact
replay pass. This is a nine-PE component result. Original-weight complete Layer0 has since passed the scoped physical contract
above. Complete dense-stage integration and full-model inference remain open. The
original export compile failure and short-budget SIM001 timeout are preserved.

The [continuous original Layer3 sequence](LAYER3-CONTINUOUS-RESET.md) now has
accepted physical positions 0 and 1 plus device reset replay. All three captures
pass the complete conditional operator audit and independent state/transport
reconstruction. The original outer supervisor exit 1 and missing COMPLETE remain
preserved; a separately admitted postcapture audit passes. Nominal CPU BF16
mismatches are 346/1,150/346 of 5,120. Bitwise parity and a propagated whole-layer
enclosure are not established. Actual 0→1→2→3 hidden flow and checkpoint restore are next; all 64 layers, full-vocabulary logits, generated tokens and
three-stage checkpoint restoration remain unqualified.

The following records retain the scope of earlier milestones.

The [complete original Layer3 static graph](LAYER3-STATIC-TRANSPORT.md) now has
accepted single-position physical and conditional operator-contract evidence.
One full epoch is captured, the runtime exits normally, and independent checks
verify all transport/raw/parameter bindings and alternate exact FMA samples.
The nominal CPU BF16 comparison has 346/5,120 mismatches, maximum absolute
error 0.00390625; propagated whole-layer enclosure and bitwise parity are false.
Continuous positions/reset, all 64 layers, full vocabulary logits and three-stage
state reload remain unqualified. The records below preserve earlier scopes.

The [full136 static transport fixture](STATIC-TRANSPORT-FULL136.md) now passes
on physical WSE-3: two identical complete captures, exact protocol and ownership
checks, normal exit, released resources and verified backup. This single epoch
has no reset and includes no neural computation. Earlier results below retain
their original scope and failure history; complete neural epochs remain zero.

The [small static transport fixture](STATIC-TRANSPORT-SMALL.md) passes complete
double-read protocol and normal-stop qualification on the workstation SDK.
Its 27 PE/3 producer scope does not qualify full 136-producer physical transport or an
original neural epoch. Prior timing and native-control failures are retained.

The [actual sender-frame checks](FULL136-SENDER-FRAME.md) traverse all136
producers at three qualified instruction boundaries, yet RUN011 still receives
the same malformed prefix. These point observations do not prove continuous
DMA or wire immutability or a unique routing cause. Release and backup are
independently verified; no transport repair or neural epoch is claimed.

The [origin word receiver](FULL136-WORD-RECEIVER.md) delivers the same malformed
prefix through qualified scalar task arguments. Strict checking still fails;
ordering and retained context differ from RUN009. Bulk origin body DMA is not
necessary for this occurrence. Release and backup are independently verified;
no repair, complete neural epoch or CSL model generation is claimed.

The [body-completion boundary diagnostic](FULL136-BODY-BOUNDARY.md) detects the
same malformed prefix before the current tail receive is armed. Its strict
protocol check still fails. Comparable records, normal exit, release and backup
are independently accepted; earlier boundary, routing, DMA and sender-lifetime
alternatives remain. Complete neural epochs and CSL model generations remain zero.

A new [136-producer first-fault capture](FULL136-FIRST-FAULT.md) independently
preserves the original strict failure: all 13 earlier captures are byte-identical,
while two added fault-bank reads agree. The bad eight-word prefix is visible at
a phase 3 ordinary-tail failure; its remaining RX words are retained prior data.
Normal exit, all-process release and durable backup are verified. This is accepted
diagnostic evidence, not a protocol pass or root-cause proof. Complete neural
layer epochs and CSL model generations remain zero.

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
with ordinary value3 at the expected control-tail position. A smaller matched
[three-PE two-source comparison](NATIVE-CONTROL-MULTISOURCE.md) now passes on the
simulator and physical WSE-3, including both causal first-packet overlap witnesses.
It does not repair or explain the larger failure. A subsequent physical
[bidirectional fixture](NATIVE-CONTROL-BIDIRECTIONAL.md) accepts 200 exact words,
two synthetic rows and final stability, while its strict dual-first-send witness
fails (30/31 checks). Full simulator 012/014/015 fail before compute;013 passes
only a shortened prefix. None repairs the 136-producer fixture.

The prepared-launch physical RUN007 subsequently passes all 31 checks, including
both first-TX lease witnesses, with the same strict data/bank/row/stability checks.
Its only device change moves the existing GO after origin prepares REQUEST0.
The prior 006 and simulator failures remain preserved; one pass establishes no
unique root cause or full 136 repair.

| Component | Current accepted scope |
|---|---|
| Original Layer3 continuous execution | Physical positions 0 and 1 plus reset0; all conditional operator gates pass, original outer failure preserved; full model remains open |
| Original Layer3 static transport | One physical position and conditional operator contract accepted; nominal output differs at 346/5120 BF16 values; full model remains open |
| Full136 static transport | Physical 534 PE fixture passes full double reads and exact ownership checks; one epoch, no reset, zero neural epochs |
| Small static transport | SDK27 PE/3 producer fixture passes both full reads, exact provenance/release checks and normal exit; full 136-producer physical and neural inference remain unqualified |
| Full136 actual sender-frame checks | RUN011 strictly fails after all136 producer three-point checks; zero sender guard faults; same retained fault context, different origin order; no continuous DMA or wire proof |
| Full136 origin word receiver | RUN010 strictly fails through scalar IQ arguments; same bad prefix with different ordering/context; bulk origin body DMA not necessary, upstream cause unresolved |
| Full136 body-completion boundary | RUN009 strictly fails with the same invalid prefix before current tail arming; independently accepted boundary evidence, release and backup; no unique root cause or repair |
| Full136 first-fault diagnostic | Physical RUN008 strictly fails; stable completed fault bank, 408 exact producer records, 13 prior captures byte-identical, normal exit/release and backup independently verified; no root-cause or neural-inference pass |
| Native bidirectional lifecycle | Physical 007: all 31 checks, 200 words,two rows,full banks/stability and both first-TX witnesses pass; origin RX during unfinished TX observed; normal exit/release. Physical006 overall 30/31 failure and simulator failures preserved |
| Native control two-source comparison | Same three-PE program, four packets and73 exact words pass independent simulator and physical checks; complete banks/suffixes, both first-lease witnesses and stable state; normal exit/release, larger fixture still open |
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
