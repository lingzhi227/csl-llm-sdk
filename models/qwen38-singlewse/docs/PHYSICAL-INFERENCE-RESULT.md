# Initial full-model functional milestone on one WSE-3

The complete original **Qwen3.8-27B-FP8 text network generated two continuous
sentences and EOS on one physical WSE-3**. This closes the initial operator
implementation and functional demonstration phase. Communication optimization,
faster inference and broader numerical qualification remain future work.

This release does **not** claim strict CPU numerical parity, complete numerical
certification, completed reset/replay qualification, or optimized performance.
The original failed comparisons remain unchanged. The user explicitly requested
publication at this functional milestone after the final small norm diagnostic.

## Observed physical execution

Prompt: “Write two short sentences about a sunny morning, mentioning warm
sunlight and birds singing.” Thinking was disabled in the official template.

> The warm sunlight bathed the garden in a golden glow, waking up the dew on the leaves. Nearby, birds began their cheerful morning chorus, filling the air with lively melodies.

The captured request used 28 prompt tokens and 37 generated tokens including EOS:
64 processed positions, each through all 64 layers and the full 248,320-entry
vocabulary head. The host uploaded prompt IDs once; every dependent next-token ID, hidden state,
logit and recurrent update was computed on wafer. No CPU reference output was
fed into the physical inference.

| Item | Recorded result |
|---|---|
| Physical attempt | `resident-generation-hw-003` |
| Qualified full compile | `full-resident-hw-004` |
| Model revision | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` |
| Layers | 64: 48 Gated DeltaNet and 16 full-attention layers |
| Original text tensors | 1,251; 29,468,003,328 original payload bytes |
| Weight residency | 849,313 PEs; initialization once, no per-token weight upload |
| Complete application | 870,000 PEs, 434 compiled programs; all mappings checked |
| Maximum per-PE admission | 43,728 bytes plus 4,096 stack reserve = 47,824, below 48,128 |
| Context capacity | 96 positions; this request used 64 |
| Endpoints | All 870,000 checked after the completed request |

The stack allowance is not a measured dynamic stack peak. Initialization read
back 2,249 original-weight samples (all small tensors and matrix strip endpoints)
and all duplicated actor parameter copies. It did not exhaustively read every
matrix PE after generation. Text-only execution excludes unused vision/MTP
branches; the text network is complete.

## Actual measured latency

| Measurement | Seconds |
|---|---:|
| Original-weight initialization, separate from request |396.737784 |
| Prompt to first generated ID (TTFT) |838.693552 |
| Mean dependent step |30.078046 |
| Dependent step range |30.012121–30.139299 |
| Complete request through EOS |1921.589063 |

Dependent decode throughput was 0.033244
tokens/s, counting EOS. These measurements include device calls and host ID
readback. Diagnostic capture was enabled. No corrected capture-disabled
sentence measurement completed, so these are not production throughput or
peak WSE-3 figures. The current controller serializes extensive communication;
communication and scheduling are the next performance targets.

## Numerical evidence and its limits

Earlier operator experiments qualified FP8 quantization/GEMV, BF16 storage/GEMV,
normalization/nonlinear operations, recurrent state, attention and communication
on their recorded fixtures. They do not prove every full-model input.

A physical RMSNorm accuracy experiment covered 4,290 actual captured inputs,
21,964,800 outputs per method. Compensated 5120-wide square accumulation reduced
differences from directly rounded FP64 from 3,143 to 239, with maximum 1 BF16 ULP.
Only 5120-wide normalization was changed; 256-wide Q/K and 128-wide GDN norms were
unchanged. The complete compile and sentence above use this corrected source.

The final small check replayed 65 saved position-zero norm inputs (332,800 scalar
outputs), with original gains, against the unchanged PyTorch expression:

| Comparison | Different values | Maximum BF16 ULP |
|---|---:|---:|
| Original physical serial sum vs PyTorch |32 |1 |
| Corrected physical sum vs PyTorch |3 |1 |
| PyTorch vs directly rounded FP64 |4 |1 |
| Corrected physical sum vs directly rounded FP64 |3 |1 |

At position 0, the third layer's input normalization matched PyTorch exactly
after correction. This is a local rounding diagnostic on the recorded operands;
it does not explain or certify every full-model difference.

**The frozen full-prefix CPU comparison did not pass.** Before the user closed
the phase, it recorded 2330 of the expected 4224
vector comparisons: 35 complete positions and 20 layers of the next position.
The first threshold failure was position 0/layer 21 (zero-based). Maximum observed
relative L2 was 22.735748% for layer
outputs, 22.843094% for final
normalization, and 21.829447% for logits.
The full 64-position reference evaluation was stopped at the user's phase closure.
No thresholds or historical results were changed to turn this into a pass.

Different reduction orders and quantization boundaries can cause differences
across implementations; [PyTorch documents this limitation](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html).
The local evidence supports ordinary rounding differences at the tested norm
boundaries. It does not establish that all observed full-chain differences are
harmless. Fluent text alone is not a complete correctness proof.

## Release and retained evidence

The first request completed, including EOS, endpoint checks and full captured
outputs. The later reset/measurement requests were stopped after the strict
comparison failed; this corrected session therefore **did not stop normally**
and its full four-request acceptance did not complete. Its owned runtime job
was cancelled and released with no cleanup errors. All 39 physical jobs from
this development phase have released their systems. The independent reference
was also stopped, retaining its partial comparisons and failure record.

[Milestone receipt](../evidence/functional-milestone-001/COMPLETE.json),
[physical generation](../evidence/resident-generation-hw-003/request-00/generation.json),
[resource release](../evidence/resident-generation-hw-003/run-audit.json),
[strict comparison summary](../evidence/reference-sequence-002/comparison-summary.json),
[small norm diagnostic](../evidence/norm-rounding-diagnostic-001/COMPLETE.json),
[original strict contract](ACCEPTANCE.md), and
[reproduction guide](REPRODUCING-FULL-INFERENCE.md).

Artifact SHA-256: `a637f385554ba8d6b186be74a31f0d449215323b64d96955b3c51534132213da`.
Device source is byte-identical to that compiled artifact's source manifest.
Weights, binaries, SDK files and raw arrays remain on the execution hosts;
public sources, hashes and compact receipts describe their provenance.

## Next work

1. Reduce serialized whole-wafer communication and unnecessary data movement.
2. Improve matrix-block parallelism and device scheduling; measure end-to-end
   TTFT and decode again after each qualified change.
3. Complete clean reset/replay and capture-disabled measurements, then expand
   numerical investigation and prompt/context coverage. Preserve this baseline.

An uncompiled performance experiment was parked and is excluded from the public
published implementation. No performance improvement from it is claimed.
