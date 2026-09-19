# Implementation status

September 19, 2026 UTC: the latest original layer 3 physical diagnostic records 20 of 24
attention READY heads, 549 of 576 Q/K/V packet observations and 380 FIFO events.
The run stopped normally and all owned resources were independently released.
Complete original neural epochs and complete CSL model generations remain zero.

| Component | Current accepted scope |
|---|---|
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

The immediate progress proposal is device-only, receiver-confirmed Q/K/V sender
credits. A moved trace is an alternative diagnostic. Neither is currently
implemented or qualified. The overwritten QK RMS intermediates must be handled by
an explicitly qualified enclosure gate or a retained-evidence design before a
complete mathematical capture; no simulated values may be labeled observations.

[Current diagnostic](LAYER3-FIFO-TRACE.md) · [Full MLP](HW01-FULL-MLP.md) ·
[CPU reference](FULL-REFERENCE.md) · [Earlier milestones](MILESTONES.md)
