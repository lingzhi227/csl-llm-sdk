"""Bind full compact2D physical correctness, communication timing and release evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--physical',required=True);p.add_argument('--simulation',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
evidence=ROOT/'evidence';inputs={}
def read(name,file):
    path=evidence/name/file;raw=path.read_bytes();inputs[str(path.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest();return json.loads(raw)

sim=read(a.simulation,'COMPLETE.json');assert sim['result']['passed'] and sim['result']['normal_stop'] and not sim['physical']
assert read(a.simulation,'workstation-release.json')['workstation_released']
completed=read(a.physical,'COMPLETE.json');result=read(a.physical,'result.json');fixture=read(a.physical,'fixture.json')
reference=read('full-bf16-matrix-hw-003','result.json');plan=read(a.physical,'source/region.json');audit=read(a.physical,'source/flow-audit.json')
assert audit['passed'] and completed['all_owned_jobs_released'] and result['passed'] and result['physical'] and result['normal_stop']
for gate in ['all_native_dots_subtrees_gathers_and_outputs_exact','all_packet_and_callback_counters_exact','all_loaded_weights_retained','full_background_retained','original_target_weights_retained']:assert result[gate]
assert len(result['cases'])==6 and [c['case'] for c in result['cases']]==list(range(6))
assert result['fixture_sha256']==sim['result']['fixture_sha256']==reference['fixture_sha256']=='b9fc3fa2e6150f2de1e017cc5f0490ec4374ca527e25bc1e22e4250e9f646f1c'
art=read(a.physical,'artifact.json');sram=read(a.physical,'sram.json');assert sram['passed'] and sram['application_pes']==1025
jobs=[];stages={}
for phase in ['compile','run']:
    stage=read(a.physical,phase+'-audit.json')
    assert stage['exit_code']==0 and stage['error'] is None and not stage['cleanup_errors'] and not stage['uncorrelated_new_jobs']
    assert stage['jobs'] and all(j['phase']=='SUCCEEDED' and j['released'] for j in stage['jobs'])
    jobs+=stage['jobs'];stages[phase]=stage.get('elapsed_seconds',stage.get('seconds'))
matched=[]
for c in range(6):
    old=reference['cases'][c];new=result['cases'][c];assert old['kind']==new['kind'] and old['mode']==new['mode']
    before=old['source_send_to_all_outputs_return_cycles'];after=new['source_send_to_all_outputs_return_cycles']
    matched.append(dict(case=c,kind=new['kind'],mode=new['mode'],complete_original_matrix=new['complete_original_matrix'],reference_cycles=before,candidate_cycles=after,speedup=before/after,latency_reduction_fraction=1-after/before))
max_sram=max(r['low_section_end']+r['stack_allowance_bytes'] for r in sram['records'])
summary=dict(schema='wse-compact-projection-summary-v1',passed=True,physical=True,full_model=False,full_model_speed_target_achieved=False,
             model=fixture['model'],revision=fixture['revision'],matrix_shape=[48,5120],tile_shape=[2,128],
             physical_application=plan['application'],workers=960,input_owners=40,physical_attempt=a.physical,simulation_attempt=a.simulation,
             artifact_sha256=art['sha256'],fixture_sha256=result['fixture_sha256'],matched_cases=matched,
             median_matched_bf16_speedup=statistics.median(x['speedup'] for x in matched if x['kind']=='bf16'),
             native_output_values_checked=6*960*2,subtree_output_values_checked=6*960*2,returned_output_values_checked=6*48,
             input_packet_words_checked=6*960*65,retained_bank_bytes=960*8814*4,
             original_bank_bytes=sum(w['fp8_slots']*260+w['bf16_slots']*512 for w in fixture['workers']),
             sram_with_stack=max_sram,sram_margin=48128-max_sram,compiled_application_images=sram['compiled_elf_images'],all_owned_jobs_released=True,jobs=jobs,
             static_max_input_hops=audit['max_input_hops'],static_max_link_words_including_teardown=audit['max_link_words_including_teardown'],
             host_resident_upload_alias_fence_seconds=result['host_initialization_seconds'],
             initialization_note='This upload/alias/fence interval starts after SDK runtime is ready; artifact upload, scheduling and runtime startup are excluded.',
             timing_boundary='Controller kick through complete48-output delivery, includes40 distributed K input transfers/native/reductions/gather. P17 starts with the same input at one controller; current input is pre-positioned at40 ingress owners. Both exclude prior host operand placement/arm; one physical SDK host channel. Early-finished PE validation copies may overlap remaining work; no time is subtracted.',
             numerical_comparison='Bit-identical P17 original fixture, full bank payloads, all inputs and the same ordered FP32/independent FP64 oracle. Only coordinate metadata changes; the supplementary FP8 case remains representative smoke, not a whole original FP8 projection.',
             limitation='Component-only compact coordinates, host-provided initial operands and centralized48-row qualification sink. Main matrix shapes, adjacent consumers, production scheduler/state placement, full model/token feedback and>=2000dependenttps are not qualified.',
             evidence_sha256=inputs)
with a.output.open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['evidence_sha256','jobs']}))
