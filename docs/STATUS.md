# Implementation status

September 19, 2026 UTC: the 45-PE native KV/Q SDK fixture independently passed
three operations, reset, all raw payload/canary checks and 525 invalid API calls.
Its accepted binaries were reused; all thirteen captures survived as immutable
files and the run stopped and released normally. The latest original layer-3
physical diagnostic still records 20 of 24 READY heads and 549 of 576 Q/K/V
packet observations. Complete original neural epochs and CSL model generations
remain zero.

| Component | Current accepted scope |
|---|---|
| Native KV/Q ownership | 45-PE SDK fixture, three operations/reset, 29,184 exact payload halfwords, 192 canaries, 525 actual API rejects, normal stop and release |
| Original layer 3 MLP | HW01: complete 5120→17408→5120 MLP, four original/changed/zero inputs, full resident weight retention and independent numerical audit on physical WSE-3 |
| Full text CPU reference | Original 64 layers, 851 text tensors, full vocabulary, four generated tokens and complete original cache restoration; CPU acceptance only |
| Full layer 3 compiled graph | Latest diagnostic 563 programs, 33750 PEs, 30576 matrix PEs; maximum ordinary storage plus 4096 stack 47920 under 48128 |
| Full layer 3 physical diagnostic | Twenty READY heads; incomplete heads 17, 21, 22 and 23 lack K/V fanout data; no complete neural epoch or reference numerical acceptance |
| Finite FIFO observation | SDK16-PE capacity 64/128 and device-gated drain fixture; physical six-source finite-prefix capture through 50 idle locations |
| Q/K/V synthetic fanout |112 roots, 24 heads, 576 packets, 192 rows and 24576 exact markers completed in the compressed SDK graph |
| Dense-stage transport | Physical 600-PE two-epoch ownership, packed BF16 bank and connected transport qualification |
| Selected original attention | WP15: selected query/KV head, two original inputs, persistent KV and independently checked handoffs/numerical stages |
| Recurrent primitives | Scoped convolution/DeltaNet/gated-normalization simulator milestones; complete original recurrent layer remains integration work |
| WP09 recurrent integration | Unaccepted; prior lifecycle/weight-readback issue remains open |
| Three sequential logical stages | Target layers 0–19, 20–43, 44–63 with host hidden/KV/DeltaNet/convolution checkpoints; complete device execution not yet qualified |
| Complete CSL text generation | Not yet accepted |

The current native path uses four independent KV streams per receiver and a
bounded Q packet window. The synthetic transport fixture is accepted; complete
original head/root SRAM fit, a full graph compile and original-layer numerical
acceptance remain separate requirements. The overwritten QK RMS intermediates
need an applicable enclosure gate or retained-evidence design; reconstructed
intermediates must not be labeled observations.

[Native SDK milestone](NATIVE-KV-SDK.md) · [Physical diagnostic](LAYER3-FIFO-TRACE.md) ·
[Full MLP](HW01-FULL-MLP.md) · [CPU reference](FULL-REFERENCE.md) ·
[Earlier milestones](MILESTONES.md)
