# Resident native loops and direct fused consumers

P23 makes the P22 disjoint stage map concrete at native-loop, original-weight
address and arithmetic-body level. It is a compiler/storage milestone, not a
connected layer execution. Full internal routes, distributed reductions, GDN
preprocessing and layer0 -> layer1 numerical execution remain unfinished.

## Native ownership inside resident stages

`spatial/layer_schedule.py` keeps all66 stage rectangles and all498 original
matrices. Complete contraction groups give each PE a fixed K slice while it
loops over its local output rows. The short final group retains a small fixed
set of K slices. Each invocation holds only one partial output block, released
after both local send completion and consumer credit. It does not allocate an
array of every partial output on a nearly full weight PE.

The independent census checks194 regions,115077120 native tiles, exact original
matrix shapes and every PE's bank counts. Maximum matrix payload is35100bytes;
the least spare auxiliary capacity is1052 pages per region. Original BF16
matrices use1x128 rows. Original FP8 matrices retain256 weight bytes per native
tile plus an exact FP32 expansion of the original BF16 scale. Repacking changes
neither original weight bits nor quantization blocks.

MLP gate/up use8x32 tiles with paired output roots; down uses4x64. The choice
screens feasible shapes using the frozen physical P17 decode-plus-native tile
cycles. For a GDN layer, the most loaded gate/up PE has134 calls and a conditional
76553.78 native busy cycles; down has135 calls and79700.63 cycles. Corresponding
2x128 native-loop candidates cost104034.25 and104810.63 cycles. These are reused
primitive costs, **not measurements of the new loops or stage latency**. They
exclude new loop overhead, reductions, input distribution, quantization, GDN,
BF16 work, credits and token feedback. No clock or TPS is inferred.

## Cross-kernel fusion and bounded ownership

`csl/layer_native.csl` retains native input packets for the matrix row loop.
Actual full-K reduction must finish before root rounding. The compile shell
exports a rounding hook but does not implement that reduction; rounding a local
partial would be incorrect and is not accepted execution.

The paired full-K gate/up BF16 values go directly to the resident
`csl/layer_fusion.csl` actor. It performs the original BF16 SiLU and multiply,
waits for all128 original group values, computes the same FP32 reciprocal-
multiply scale, and emits two64-column native down packets. There is no required
17408-value centralized activation gather. Groups that straddle output roots
must supply all tagged pairs; routes for that join remain to be generated.

Each selected actor coexists with16640bytes of original matrix payload. There
are52 such actors in a GDN stage and128 in an attention stage; they cover all136
quantization groups in increasing group order. Later requests cannot overwrite
mutable actor scratch. Every outgoing fragment remains owned until both send
completion and downstream consumption. Headers, queues and physical credit
routes are still open integration work.

GDN state now has explicit4-key by8-value FP32 pages, so no PE needs a contiguous
4KiB hole. Two requests have49152 distinct128-byte pages per GDN layer. The
`gdn_page.csl` body provides decay/prediction/update over one page. The complete
32-key-page reductions, original convolution/gates/norm and request-specific
state commit must be connected before it represents one recurrent head.

One request controller occupies a proved unused PE in every stage. Replicating
the complete stage lease guard in every full bank exceeded SRAM in attempt004.
Workers keep only local epoch/input/row/send-credit ownership. Controller calls
carry explicit slot and lease identifiers, so interleaved requests cannot use
an implicit last-request slot. Compiling the guard does not prove distributed
reset, liveness or physical backpressure.

## Exact validation scope

`layer-backend-compile-008` compiles six selected role profiles with the SDK2.10.1
WSE-3 compiler and checks actual ELF sections plus a declared4096-byte stack:

| Role | Payload bytes | SRAM including stack | Remaining bytes |
|---|---:|---:|---:|
| GDN mix + FP8 + BF16 row + state page | 35256 | 47712 | 416 |
| Paired8x32 gate/up + projection rounding | 35256 | 47200 | 928 |
| 4x64 down | 35256 | 46560 | 1568 |
| Attention mix FP8 shell | 35256 | 46656 | 1472 |
| Native gate/up + fused activation/quantization | 16640 | 32080 | 16048 |
| Stage request controller | 0 | 10144 | 37984 |

These are selected profiles on a6-PE compilation, not all actual stage roles or
complete-layer SRAM admission. The4096-byte stack is a reservation, not a measured
dynamic peak. No fabric, full-layer scheduler or host-to-device model loader is
part of this shell. Attention-specific state arithmetic is not present. The live
BF16-row source differs from compiled008 only in its provenance comment; frozen
source hashes remain exact and separate.

Attempts001/002/007 retain their compiler syntax failures. Attempt003 retains the
LLVM pthread failure; explicit CPU affinity was restored, without claiming the
underlying cause proved. Attempt004 compiled but exceeded the declared48128-byte
ceiling by up to1072bytes. Attempts005 and006 admitted earlier smaller compositions;
they do not replace008's scope. No failed snapshot was overwritten.

`layer-weight-audit-001` verifies original publisher shard hashes and samples
6080 packed tiles across all16 matrices of layers0/1, including original row/scale
boundaries and matrix ends. It independently checks the scalar packer against
bulk original-array transposes and verifies all12 small parameters. This is
sampled packing qualification, not exhaustive packed-bank/device readback or
neural execution. Schedule002 changes only explanatory epilogue text relative to
the audited selected-layer schedule001; addresses are unchanged.

All97 source tests pass. They include full-model bank/shape coverage, ragged-tail
K coverage, gate/up ownership,49152-page request isolation, packing boundaries and
the previously delivered predictor's11 tests. `pipeline-prediction-002` remains
bound to the older P22 cyclic2x128 schedule. It does not predict the new retiled
loops. Its finite-request service model rejects missing costs/clock provenance;
full-model TPS is unavailable. See PERFORMANCE-PREDICTION.md.

All eight compiler services and the original-weight audit have invocation-bound
release receipts. No new WSE jobs were submitted. The final ALCF snapshot has
no own active jobs or system assignments; provider billed node-hours are not
available from that snapshot. Initialization payloads and ELFs remain remotely
stored; the publication contains source and compact metadata only.

The next acceptance boundary remains the real connected layer0 -> layer1 path:
actual operand multicast, full-K reductions, GDN preparation/state, RMS and fused
MLP, direct successor consumption, two isolated requests and warm reset. Only
then can complete service costs guide further stage balancing and full64-layer
feedback. The >=2000 aggregate generated tokens/s target remains unmet.
