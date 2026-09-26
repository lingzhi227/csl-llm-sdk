"""Write the publication result from successful receipts, never from progress."""
import argparse
import re
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--physical',required=True);args=parser.parse_args()
assert re.fullmatch('resident-generation-hw-[0-9]{3}',args.physical)
folder=ROOT/'evidence'/args.physical/'acceptance'
r=json.loads((folder/'COMPLETE.json').read_text())
assert r['passed'] and r['full_model_complete'] and r['physical']
physical=r['physical_attempt'];reference=r['reference_attempt']
assert physical==args.physical
reuse=json.loads((folder/'physical/reuse-artifact.json').read_text());compiled=Path(reuse['compiled_from']).name
sram=json.loads((folder/'physical/sram.json').read_text())
assert reuse['artifact_sha256']==r['artifact_sha256']==sram['artifact_sha256']
numeric=json.loads((ROOT/'evidence'/reference/'comparison-summary.json').read_text())
assert numeric['records']==r['numerical_comparisons'] and numeric['first_failure'] is None
assert all(v['all_passed'] for v in numeric['by_kind'].values())
candidate=json.loads((folder/'physical/CANDIDATE.json').read_text())
first=candidate['runs'][0];positions=first['processed_positions'];tokens=len(first['generated_ids'])
rows=[]
for run in r['measured_capture_disabled_requests']:
    steps=run['dependent_step_seconds']
    rows.append(f"| {run['request']} | {run['ttft_seconds']:.3f} | {statistics.mean(steps):.3f} | "
        f"{min(steps):.3f}–{max(steps):.3f} | {run['end_to_end_seconds']:.3f} | "
        f"{run['dependent_decode_tokens_per_second_including_eos']:.5f} |")
errors=[]
for kind,v in numeric['by_kind'].items():
    errors.append(f"| {kind} | {v['checks']} | {v['max_relative_l2']:.8g} | {v['min_cosine']:.8g} | "
        f"{v['max_max_abs_error']:.8g} | {v['max_rms_abs_error']:.8g} |")
text=f'''# Complete physical Qwen3.8 text inference

The complete original Qwen3.8-27B-FP8 text network generated two continuous
sentences on one physical WSE-3. All 64 layers and all 248,320 vocabulary entries
executed on wafer. Frozen full-vector numerical validation, device token
selection, EOS, reset/replay and normal resource release passed.

The accepted physical attempt is `{physical}`, using compile
`{compiled}` and independent sequential reference
`{reference}`. Development stopped after this functional acceptance
at the user's request; the uncompiled optimization draft is excluded.

## Workload and output

Prompt: “Write two short sentences about a sunny morning, mentioning warm
sunlight and birds singing.” The official chat template disabled thinking.
The prompt contains {len(first['prompt_ids'])} tokens; generation produced
{tokens} tokens including EOS, for {positions} processed positions.

> {r['generated_text']}

The host uploads the prompt once and triggers dependent calls. Every next-token
ID feeds embedding on wafer; hidden vectors, logits, recurrent updates and
next-token IDs are never uploaded from a CPU reference. All four requests use
the same once-loaded weights. The greeting and both capture-disabled sentence
requests execute after clean device resets. Both sentence replays have the
same IDs as the captured request.

## Numerical qualification

Every one of the {positions} processed positions was checked: 64 complete layer
vectors, final normalization and all vocabulary logits. That is
{r['numerical_comparisons']:,} vector comparisons and
{positions*(64*5120+5120+248320):,} scalar pairs. This is full-vector comparison,
not a sampled hidden-value check. Exact BF16 equality is not required; the
thresholds and BF16 selection rule were frozen before candidate observation.

| Output | Vector checks | Maximum relative L2 | Minimum cosine | Maximum absolute error | Maximum RMS error |
|---|---:|---:|---:|---:|---:|
{chr(10).join(errors)}

All generated IDs are the exact lowest-ID argmax of the captured device logits.
The CPU reference winner is required outside the frozen BF16 tie band. The
reference uses the same original FP8 checkpoint and one-position-at-a-time
cache semantics; the earlier chunk-prefill reference is separate evidence.

The earlier physical attempt002 produced readable sentences but failed the
frozen sequential numerical checks. Original-operator replay localized its
initial divergence to serial FP32 accumulation in5120-wide RMSNorm. The final
code uses compensated accumulation at that width, first qualified physically
on4290 captured inputs and then by the complete comparisons above. The frozen
criteria were not relaxed. Failed attempt002 and its diagnostic receipts are
retained alongside the successful result.

## Measured latency

These are actual measurements of this unoptimized CSL implementation. They
include the device call and host ID readback; they are not estimates of peak
WSE-3 speed or timings from a partial operator. Capture-disabled runs are the
headline measurements. Decode throughput below includes the final EOS token.

| Request | TTFT (s) | Mean dependent step (s) | Step range (s) | End-to-end (s) | Decode tokens/s |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Runtime/artifact initialization took {r['initialization_seconds']:.3f} seconds;
original-weight initialization took {r['weight_initialization_seconds']:.3f}
seconds. Both are excluded from request timing. The capture-enabled validation
request took {first['ttft_seconds']:.3f} seconds to first ID and
{first['end_to_end_seconds']:.3f} seconds end to end. Its diagnostic overhead is
reported separately from the table.

The current scheduler serializes many operand broadcasts and returns. This
functional release does not claim that the implementation is performance
optimized. The two measured replays do not constitute a broad throughput study.

## Residency, capacity and limitations

- Original model: `Qwen/Qwen3.8-27B-FP8` at
  `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`.
- All 1,251 text tensors, 29,468,003,328 payload bytes, loaded into 849,313 weight
  PEs. No layer/expert pruning, weight reload between tokens or CPU neural
  fallback. Text-only inference does not use the vision/MTP branches.
- {sram["application_pes"]:,} application PEs; {sram["application_elfs"]} compiled programs cover every application PE.
  Maximum ordinary section end plus declared 4 KiB stack reserve is {sram["max_low_section_end"]+sram["stack_allowance"]:,}
  bytes, below the {sram["ceiling"]:,}-byte ceiling. The reserve is not a measured peak stack usage.
- Context capacity is 96 positions. This workload is not long-context,
  multimodal, corpus-quality or arbitrary-prompt validation.
- Initialization checks 2,249 original weight samples (including complete small
  tensors and matrix strip endpoints) and all actor parameter copies. It does
  not perform exhaustive post-generation matrix-weight readback.
- All 870,000 endpoint traces were checked after each request. The SDK runtime
  stopped normally; its invocation-correlated physical job succeeded and
  released the system.

## Evidence and reproduction

[Final acceptance](../evidence/{physical}/acceptance/COMPLETE.json),
[frozen criteria](../configs/full-acceptance-v1.json),
[numerical summary](../evidence/{reference}/comparison-summary.json),
[reproduction guide](REPRODUCING-FULL-INFERENCE.md) and
[publication provenance](../SOURCE_EXPORT.json).

Artifact SHA-256: `{r['artifact_sha256']}`.
Criteria SHA-256: `{r['acceptance_sha256']}`.
Raw tensors and compiled payloads are retained remotely; public JSON receipts
alone cannot replay all numerical comparisons.
'''
(ROOT/'docs/PHYSICAL-INFERENCE-RESULT.md').write_text(text)
readme=f'''# Complete Qwen3.8 inference on one WSE-3

**Qwen3.8-27B-FP8 generated two continuous sentences using its complete original
64-layer text network on one physical WSE-3.** Full-vector numerical validation,
EOS, device-only dependent token feedback, reset/replay and normal release passed.

> {r['generated_text']}

The run used {len(first['prompt_ids'])} prompt tokens and {tokens} generated
tokens including EOS. All 1,251 text tensors remain resident and the complete
248,320-entry vocabulary head executes on wafer. The initial context capacity
is 96 positions; unused vision and MTP branches are outside this text workload.

Read the [physical result and measured timings](docs/PHYSICAL-INFERENCE-RESULT.md),
[acceptance](evidence/{physical}/acceptance/COMPLETE.json),
[frozen numerical criteria](configs/full-acceptance-v1.json),
[reproduction guide](docs/REPRODUCING-FULL-INFERENCE.md) and
[publication notes](PUBLICATION.md).

Device code is in `csl/` and `resident/`; `runtime/` contains original-weight
loading and bounded physical execution, `reference/` the independent CPU
oracle, and `tools/` the placement/build/staging/evidence utilities. The complete
layout and current generated CSL match the actual physical compile byte for
byte. Model weights, binaries and raw arrays are not distributed.

This is a functional release with measured, unoptimized latency. Development
stopped after correctness acceptance at the user's request. It is distinct
from the historical 20-layer Qwen stage and from the GPT-OSS two-token release.
'''
(ROOT/'README.md').write_text(readme)
print(json.dumps(dict(report_written=True,physical_attempt=r['physical_attempt'],numeric_checks=r['numerical_comparisons'])))
