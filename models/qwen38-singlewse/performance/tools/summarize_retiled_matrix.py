"""Bind matched-input, numerical, physical timing and release evidence for P18."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--physical',required=True);p.add_argument('--simulation',required=True);p.add_argument('--comparison',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
evidence=ROOT/'evidence';inputs={}
def read(name,file):
    path=evidence/name/file;raw=path.read_bytes();inputs[str(path.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest();return json.loads(raw)

simulation=read(a.simulation,'COMPLETE.json');assert simulation['result']['passed'] and simulation['result']['normal_stop'] and not simulation['physical']
assert read(a.simulation,'workstation-release.json')['workstation_released']
comparison=read(a.comparison,'result.json');assert comparison['passed'] and comparison['all_original_bf16_weights_identical'] and comparison['all_bf16_operands_identical']
assert read(a.comparison,'workstation-release.json')['workstation_released']
completed=read(a.physical,'COMPLETE.json');result=read(a.physical,'result.json');fixture=read(a.physical,'fixture.json')
reference=read('full-bf16-matrix-hw-003','result.json')
assert completed['all_owned_jobs_released'] and result['passed'] and result['physical'] and result['normal_stop']
assert result['all_native_dots_subtrees_gathers_and_outputs_exact'] and result['all_packet_and_callback_counters_exact'] and result['all_loaded_weights_retained']
assert result['full_background_retained'] and result['original_target_weights_retained']
assert len(result['cases'])==6 and [c['case'] for c in result['cases']]==list(range(6))
assert comparison['candidate_fixture_sha256']==result['fixture_sha256']==simulation['result']['fixture_sha256']
assert comparison['reference_fixture_sha256']==reference['fixture_sha256']
art=read(a.physical,'artifact.json');sram=read(a.physical,'sram.json');assert sram['passed'] and sram['application_pes']==1262
jobs=[]
for phase in ['compile','run']:
    stage=read(a.physical,phase+'-audit.json')
    assert stage['exit_code']==0 and stage['error'] is None and not stage['cleanup_errors'] and not stage['uncorrelated_new_jobs']
    assert stage['jobs'] and all(j['phase']=='SUCCEEDED' and j['released'] for j in stage['jobs'])
    jobs+=stage['jobs']
matched=[]
for c in [0,1,2,3,5]:
    old=reference['cases'][c];new=result['cases'][c];assert old['kind']==new['kind']=='bf16' and old['mode']==new['mode']
    before=old['source_send_to_all_outputs_return_cycles'];after=new['source_send_to_all_outputs_return_cycles']
    matched.append(dict(case=c,mode=new['mode'],reference_cycles=before,candidate_cycles=after,speedup=before/after,latency_reduction_fraction=1-after/before))
max_sram=max(r['low_section_end']+r['stack_allowance_bytes'] for r in sram['records'])
summary=dict(schema='wse-retiled-matrix-summary-v1',passed=True,physical=True,full_model=False,full_model_speed_target_achieved=False,
             model=fixture['model'],revision=fixture['revision'],matrix_shape=[48,5120],reference_tile_shape=[2,128],candidate_tile_shape=[8,32],
             physical_application=[631,2],workers=960,physical_attempt=a.physical,simulation_attempt=a.simulation,comparison_attempt=a.comparison,
             artifact_sha256=art['sha256'],fixture_sha256=result['fixture_sha256'],matched_bf16_cases=matched,
             median_matched_bf16_speedup=statistics.median(x['speedup'] for x in matched),
             fp8_submatrix_cycles=result['cases'][4]['source_send_to_all_outputs_return_cycles'],fp8_smoke_matched_comparison=False,
             native_output_values_checked=6*960*8,subtree_output_values_checked=6*960*8,returned_output_values_checked=6*48,
             input_packet_words_checked=6*960*17,retained_bank_bytes=960*8814*4,
             original_bank_bytes=sum(w['fp8_slots']*260+w['bf16_slots']*512 for w in fixture['workers']),
             sram_with_stack=max_sram,sram_margin=48128-max_sram,all_owned_jobs_released=True,jobs=jobs,
             timing_boundary='Same source input-send through complete48-row return; includes input transfer, native dots, all reductions/gather/return. Host arm excluded, initialization separately reported. Source packet has2600vs2720words for the same5120 operand codes. Physical runtime uses one SDK host channel in both cases.',
             numerical_comparison='Same original BF16 matrix and all global operands. Different tiling changes FP32 addition order; each is checked against its own frozen ordered oracle and independent FP64 bounds. Original/background byte identity checked separately.',
             limitation='Local tile shape changes only; PE region remains a631x2 strip. No distributed model-value producers, consumer feedback, full-model throughput or production SRAM admission.',
             evidence_sha256=inputs)
with a.output.open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['evidence_sha256','jobs']}))
