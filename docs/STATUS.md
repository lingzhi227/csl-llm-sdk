# Implementation status

September 19, 2026 UTC: physical synthetic QK archive/alias equivalence passed
two resets, exact raw diagnostics and final output, actual source overwrite,
61 invalid API calls and normal stop. Separately, all 24 complete native heads
passed the selected 89-position SRAM/storage check with 576 bytes minimum headroom.
The accepted native KV/Q SDK transport and older FIFO physical observations remain
preserved. Complete original neural-layer epochs and CSL model generations remain zero.

| Component | Current accepted scope |
|---|---|
| Physical QK archive/alias | Four PEs, two synthetic resets, exact960archive/256output/1280source-poison each; normal stop/release |
| Selected complete native programs | 89 original positions, all24heads/24sinks, max47552 including4096stack; other programs demoted, never executable |
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

The complete native graph still needs its own compiled-fit acceptance before
original-layer runtime and numerical checks. QK diagnostics are now archived
before source-workspace reuse; the four-PE fixture establishes that lifetime with
synthetic data. Selected fit and physical alias equivalence do not establish
fullgraph execution, original-weight numerical correctness or model generation.

[Physical archive and selected fit](QK-ARCHIVE-PHYSICAL.md) ·
[Native SDK milestone](NATIVE-KV-SDK.md) · [Physical FIFO diagnostic](LAYER3-FIFO-TRACE.md) ·
[Full MLP](HW01-FULL-MLP.md) · [CPU reference](FULL-REFERENCE.md) ·
[Earlier milestones](MILESTONES.md)
