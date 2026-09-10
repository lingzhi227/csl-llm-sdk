# WP15 original projected selected attention

Independently accepted SDK2.10.1/WSE3 simulator candidate: ten PEs process two
original5120 hidden vectors through all selected1024 Q/rawgate/K/V rows, trained
Q/K RMS/partialRoPE and persistent attention cache8. Two calls share request1,
positions0/1 and contraction IDs1/2. All numerical,frame/cache,retention and shutdown
gates passed. Simulation1187.19s,peak267.37MiB. Physical systems and complete models
remain separate integration work.

See [protocol](../../docs/WP15-PROTOCOL.md), [report](../../docs/WP15-REPORT.md),
[resource plan](resource-plan.json), [physical sequence](physical-sequence.json) and
[accepted evidence](../../evidence/wp15.json). The source code and resource recipe
retain their pre-execution proposal comments/metadata because they are the exact
accepted freeze; the separate admission and report record the executed scope.

## Reproduction inputs and bounded execution

Reuse the pinned WP13 selected Q/rawgate/K/V and trained norm slices, the accepted
WP13 projection reference and WP14 QK reference. The CPU preparer
[prepare_wp15_reference.py](../../tools/prepare_wp15_reference.py) freezes four
original-hidden cases with positions0/1/2 in one request,then new-request zero.
Run its CPU recipe only in an existing qualified environment under the unchanged
2GiB/60second CPU supervisor. Transparent official source hooks preserve the
projection/view/chunk/RMS/RoPE/eager/gate operations and dtype boundaries.

The SDK preparer [prepare_wp15.py](../../tools/prepare_wp15.py) freezes the full
source/CPU artifacts,run-local selected payloads,physical sequence and300second
compile/1200second simulator recipe. It performs no SDK work. The separate
[guarded_wp15.py](../../tools/guarded_wp15.py) requires an explicit controller
admission matching the entire candidate manifest and these deadlines. Compiler
parallelism and simulator/BLAS threads remain1. The compile entry admits all10
actual ELF footprints before simulation; no second candidate is implicit.

Public evidence omits complete trained parameters and direct static parameter
transforms,retaining dtype/shape/payload hashes. Input-dependent numerical/source,
physical cache and transport observations are lossless. The complete journal is
available through its [chunk index](../../evidence/wp15-journal-index.json). Raw
weight files,raw NPZ archives,SDK products and private execution paths stay outside
this source release. The source gate's uniform-attention limitation and the separate
actual-operand counterexample checks are explicit in the report.
