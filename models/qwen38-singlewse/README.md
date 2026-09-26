# Qwen3.8 on one WSE-3: initial functional milestone

The complete original **Qwen3.8-27B-FP8 text network generated two continuous
sentences and EOS on one physical WSE-3**: all 64 layers, all 248,320 vocabulary
entries, original resident weights and device-dependent token generation.

> The warm sunlight bathed the garden in a golden glow, waking up the dew on the leaves. Nearby, birds began their cheerful morning chorus, filling the air with lively melodies.

This release closes the initial operator-development and functional-demonstration
phase. **Strict CPU numerical alignment did not pass.** Reset/replay qualification
and corrected capture-disabled sentence measurements were not completed. The
report preserves these limits and the failed numerical evidence.

The captured request generated 37 tokens including EOS from 28 prompt tokens.
Measured TTFT was 838.69s, mean dependent decode
30.08s/token, and complete request1921.59s.
These are actual unoptimized implementation measurements with capture enabled.
Communication optimization and gradually improving inference speed are next.

- [Physical results, checks, limitations and next steps](docs/PHYSICAL-INFERENCE-RESULT.md)
- [Initial milestone receipt](evidence/functional-milestone-001/COMPLETE.json)
- [Reproduction guide](docs/REPRODUCING-FULL-INFERENCE.md)
- [Publication scope and licenses](PUBLICATION.md)
- [Original strict numerical target, not achieved in this release](docs/ACCEPTANCE.md)

Device sources are in `csl/` and `resident/`; `runtime/` contains bounded physical
execution and original-weight loading, `reference/` the independent numerical
checks, and `tools/` the build/staging/evidence utilities. Historical operator
experiments and failures remain with their original scope. Model weights, raw
captures, compiled payloads and the proprietary SDK are not redistributed.
The context capacity is 96 positions. This is the full text network; unused
vision/MTP pathways are outside the workload.
