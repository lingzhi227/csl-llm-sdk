"""Record the user-approved initial functional milestone without claiming parity."""
import datetime,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads((ROOT/path).read_text())
physical='resident-generation-hw-003';compiled='full-resident-hw-004';reference='reference-sequence-002'
p='evidence/'+physical+'/';r=read(p+'request-00/generation.json');trace=read(p+'request-00/trace-check.json')
init=read(p+'initialization.json');audit=read(p+'run-audit.json');sram=read(p+'sram.json');artifact=read(p+'artifact.json');binding=read(p+'binding.json')
source=read(p+'source-manifest.json')['files'];reuse=read(p+'reuse-artifact.json')
compile_receipt=read('evidence/'+compiled+'/COMPLETE.json')
summary=read('evidence/'+reference+'/comparison-summary.json');failure=read('evidence/'+reference+'/FAILURE.json')
norm=read('evidence/norm-rounding-diagnostic-001/COMPLETE.json');norm_full=read('evidence/norm-accuracy-hw-001/result.json')
assert r['physical'] and r['all_layers']==64 and r['vocabulary']==248320 and r['eos_reached'] and r['generated_ids'][-1]==248046
assert r['processed_positions']==len(r['prompt_ids'])+len(r['generated_ids'])-1==64
assert r['no_hidden_or_next_token_upload'] and trace['all870000_endpoints_checked'] and trace['request_calls']==37
assert init['passed'] and init['text_tensors_loaded']==1251 and init['original_weight_pes_loaded']==849313 and not init['per_token_weight_upload']
assert not audit['cleanup_errors'] and not audit['uncorrelated_new_jobs'] and len(audit['jobs'])==1
assert all(j['released'] and j['phase']=='CANCELLED' for j in audit['jobs'])
assert compile_receipt['all_owned_jobs_released'] and all(j['phase']=='SUCCEEDED' for st in compile_receipt['stages'] for j in st['jobs'])
assert artifact['sha256']==sram['artifact_sha256']==binding['artifact_sha256']==reuse['artifact_sha256']
assert sram['passed'] and binding['passed'] and sram['application_pes']==binding['application_pes']==870000
assert not failure['passed'] and summary['records']==failure['comparison_checks'] and summary['first_failure'] is not None
assert norm_full['passed'] and norm_full['physical'] and norm['cases']==65
for name in ['resident_run.py','resident_loader.py','resident_plan.py','weights.py','backend.py','bounded_client.py']:
 assert hashlib.sha256((ROOT/'runtime'/name).read_bytes()).hexdigest()==source[name]
text=failure['decoded_text'];assert text and text.count('.')>=2
steps=[x['dependent_step_seconds'] for x in r['steps'][1:]]
small={name:dict(different=sum(x[name]['different'] for x in norm['records']),max_bf16_ulp=max(x[name]['max_bf16_ulp'] for x in norm['records'])) for name in ['baseline_vs_original','compensated_vs_original','original_vs_fp64','compensated_vs_fp64']}
result=dict(phase_closed=True,publication_authorized_by_user=True,scope='Initial operator development and full-model sentence-generation functionality; further communication and speed work planned',
 physical=True,system_count=1,architecture='WSE-3',model='Qwen/Qwen3.8-27B-FP8',revision='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a',
 physical_attempt=physical,compiled_attempt=compiled,reference_attempt=reference,artifact_sha256=artifact['sha256'],
 full_model_execution_observed=True,all_layers=64,vocabulary=248320,text_tensors_loaded=1251,context_capacity=96,
 prompt_tokens=len(r['prompt_ids']),generated_tokens_including_eos=len(r['generated_ids']),generated_ids=r['generated_ids'],generated_text=text,processed_positions=r['processed_positions'],
 capture_enabled=True,ttft_seconds=r['ttft_seconds'],end_to_end_seconds=r['end_to_end_seconds'],mean_dependent_step_seconds=statistics.mean(steps),min_dependent_step_seconds=min(steps),max_dependent_step_seconds=max(steps),
 dependent_decode_tokens_per_second_including_eos=(len(r['generated_ids'])-1)/(r['end_to_end_seconds']-r['ttft_seconds']),weight_initialization_seconds=init['seconds'],
 strict_cpu_alignment_passed=False,full_sequence_numerical_qualification_complete=False,numerical_vector_comparisons_observed=summary['records'],numerical_expected_comparisons=failure['expected_comparison_checks'],numerical_summary=summary,
 physical_session_completed_normally=False,job_cancelled_after_complete_sentence_capture=True,all_owned_physical_jobs_released=True,corrected_reset_and_replay_completed=False,
 small_norm_diagnostic=small,small_norm_diagnostic_cases=65,small_norm_scalar_values=65*5120,
 user_closing_instruction='Publish the initial operator-development and functional-validation milestone; retain limitations and continue communication/performance optimization in future work.',
 recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
out=ROOT/'evidence/functional-milestone-001';out.mkdir(exist_ok=True);target=out/'COMPLETE.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n')
report=f'''# Initial full-model functional milestone on one WSE-3

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

> {text}

The captured request used 28 prompt tokens and 37 generated tokens including EOS:
64 processed positions, each through all 64 layers and the full 248,320-entry
vocabulary head. The host uploaded prompt IDs once; every dependent next-token ID, hidden state,
logit and recurrent update was computed on wafer. No CPU reference output was
fed into the physical inference.

| Item | Recorded result |
|---|---|
| Physical attempt | `{physical}` |
| Qualified full compile | `{compiled}` |
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
| Original-weight initialization, separate from request |{init['seconds']:.6f} |
| Prompt to first generated ID (TTFT) |{r['ttft_seconds']:.6f} |
| Mean dependent step |{statistics.mean(steps):.6f} |
| Dependent step range |{min(steps):.6f}–{max(steps):.6f} |
| Complete request through EOS |{r['end_to_end_seconds']:.6f} |

Dependent decode throughput was {result['dependent_decode_tokens_per_second_including_eos']:.6f}
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
| Original physical serial sum vs PyTorch |{small['baseline_vs_original']['different']} |{small['baseline_vs_original']['max_bf16_ulp']} |
| Corrected physical sum vs PyTorch |{small['compensated_vs_original']['different']} |{small['compensated_vs_original']['max_bf16_ulp']} |
| PyTorch vs directly rounded FP64 |{small['original_vs_fp64']['different']} |{small['original_vs_fp64']['max_bf16_ulp']} |
| Corrected physical sum vs directly rounded FP64 |{small['compensated_vs_fp64']['different']} |{small['compensated_vs_fp64']['max_bf16_ulp']} |

At position 0, the third layer's input normalization matched PyTorch exactly
after correction. This is a local rounding diagnostic on the recorded operands;
it does not explain or certify every full-model difference.

**The frozen full-prefix CPU comparison did not pass.** Before the user closed
the phase, it recorded {summary['records']} of the expected {failure['expected_comparison_checks']}
vector comparisons: 35 complete positions and 20 layers of the next position.
The first threshold failure was position 0/layer 21 (zero-based). Maximum observed
relative L2 was {summary['by_kind']['layer']['max_relative_l2']:.6%} for layer
outputs, {summary['by_kind']['final_norm']['max_relative_l2']:.6%} for final
normalization, and {summary['by_kind']['logits']['max_relative_l2']:.6%} for logits.
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
[physical generation](../evidence/{physical}/request-00/generation.json),
[resource release](../evidence/{physical}/run-audit.json),
[strict comparison summary](../evidence/{reference}/comparison-summary.json),
[small norm diagnostic](../evidence/norm-rounding-diagnostic-001/COMPLETE.json),
[original strict contract](ACCEPTANCE.md), and
[reproduction guide](REPRODUCING-FULL-INFERENCE.md).

Artifact SHA-256: `{artifact['sha256']}`.
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
'''
(ROOT/'docs/PHYSICAL-INFERENCE-RESULT.md').write_text(report)
(ROOT/'README.md').write_text(f'''# Qwen3.8 on one WSE-3: initial functional milestone

The complete original **Qwen3.8-27B-FP8 text network generated two continuous
sentences and EOS on one physical WSE-3**: all 64 layers, all 248,320 vocabulary
entries, original resident weights and device-dependent token generation.

> {text}

This release closes the initial operator-development and functional-demonstration
phase. **Strict CPU numerical alignment did not pass.** Reset/replay qualification
and corrected capture-disabled sentence measurements were not completed. The
report preserves these limits and the failed numerical evidence.

The captured request generated 37 tokens including EOS from 28 prompt tokens.
Measured TTFT was {r['ttft_seconds']:.2f}s, mean dependent decode
{statistics.mean(steps):.2f}s/token, and complete request{r['end_to_end_seconds']:.2f}s.
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
''')
print(json.dumps(dict(phase_closed=True,strict_cpu_alignment_passed=False,physical_attempt=physical,generated_tokens=37,report_written=True)))
