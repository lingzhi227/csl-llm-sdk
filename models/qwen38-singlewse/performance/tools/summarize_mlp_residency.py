"""Bind full-model address evidence to the bounded physical helper qualification.

The metadata audit and the two physical helper roles have different scopes.
Neither proves a complete MLP executable or a full-model speed result.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
inputs={}


def read(relative):
    path=ROOT/'evidence'/relative
    raw=path.read_bytes()
    inputs[str(path.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


plan=read('mlp-residency-plan-001.json')
assert plan['audit']['passed'] and plan['original_tensors']==1251
assert not plan['compiled_all_role_sram_admitted'] and not plan['full_model_executable']
sim=read('mlp-chunk-sim-001/COMPLETE.json')
release=read('mlp-chunk-sim-001/workstation-release.json')
assert sim['result']['passed'] and sim['result']['normal_stop'] and not sim['physical']
assert release['workstation_released']
read('mlp-chunk-sim-001/source-manifest.json')
completed=read('mlp-chunk-hw-002/COMPLETE.json')
result=read('mlp-chunk-hw-002/result.json')
assert completed['all_owned_jobs_released'] and result['passed'] and result['normal_stop'] and result['physical']
assert result['epochs']==3 and result['partial_and_final_values_checked']==1152
assert result['all_counters_exact'] and result['all_original_banks_retained']
assert result['fixture_sha256']==sim['result']['fixture_sha256']=='1479fa3c66bcdd432b74513fac527132b8e4e928d3e043178e18f79ece722fed'
assert result['measured_speedup'] is None and not result['full_model']
fixture=read('mlp-chunk-hw-002/fixture.json')
assert fixture['revision']==plan['revision']
adaptation=read('mlp-chunk-hw-002/source/driver-adaptation.json')
reuse=read('mlp-chunk-hw-002/reuse-artifact.json')
assert reuse['source_attempt']=='mlp-chunk-hw-001' and reuse['compiler_succeeded_and_released']
for attempt in ['mlp-chunk-hw-001','mlp-chunk-hw-002']:
    manifest=read(attempt+'/source-manifest.json')['files']
    for name,expected in reuse['csl_hashes'].items():
        assert manifest[name]==expected
        assert hashlib.sha256((ROOT/'evidence'/attempt/'source'/name).read_bytes()).hexdigest()==expected
compile_audit=read('mlp-chunk-hw-001/compile-audit.json')
failed_host=read('mlp-chunk-hw-001/run-audit.json')
assert failed_host['exit_code']==1 and failed_host['jobs']==[]
assert failed_host['cleanup_errors']==['No submitted job proven; cleanup is not claimed']
entry_release=read('resource-audit-p20-runtime-entry-001.json')
assert entry_release['no_owned_hardware_allocation']
run_audit=read('mlp-chunk-hw-002/run-audit.json')
jobs=[]
for stage in [compile_audit,run_audit]:
    assert stage['exit_code']==0 and stage['error'] is None and not stage['cleanup_errors'] and not stage['uncorrelated_new_jobs']
    assert stage['jobs'] and all(j['phase']=='SUCCEEDED' and j['released'] for j in stage['jobs'])
    jobs.extend(stage['jobs'])
assert len({j['id'] for j in jobs})==2
artifact=read('mlp-chunk-hw-002/artifact.json')
assert artifact['sha256']==reuse['artifact_sha256']
sram=read('mlp-chunk-hw-002/sram.json')
assert sram['passed'] and sram['application']==[3,3] and sram['application_pes']==9
assert {Path(r['file']).name:r['sha256'] for r in sram['records']}==reuse['elf_hashes']
roles={}
for name,xy in [('header',[5,1]),('collector',[5,2])]:
    records=[r for r in sram['records'] if any(z[:2]==xy for z in r['rectangles'])]
    assert len(records)==1
    r=records[0];total=r['low_section_end']+r['stack_allowance_bytes']
    roles[name]=dict(original_embedding_bytes=34816,allocated_bank_bytes=35256,
                     low_section_end=r['low_section_end'],stack_allowance=r['stack_allowance_bytes'],
                     total=total,margin=48128-total)
assert roles['header']['total']==46128 and roles['collector']['total']==46000
account=read('resource-audit-p20-final-001.json')
assert account['no_owned_hardware_allocation'] and not account['own_active'] and not account['own_assignments']
summary=dict(schema='wse-mlp-residency-and-chunk-summary-v1',model=fixture['model'],revision=plan['revision'],
             metadata_audit=plan['audit'],physical_result=result,physical_application=[3,3],
             physically_qualified_embedding_helpers=2,compiled_images=sram['compiled_elf_images'],helper_sram=roles,
             chunk_words=8,own_buffer_bytes=512,child_credit_buffer_bytes=32,
             numerical_order='Two ordered FP32 joins: (A+B)+C. Three synthetic partial-vector cases including cancellation, zero and warm nonzero replay; original down projection is not executed.',
             bank_scope='Both physical helpers retain the same68 original embedding tiles in row-major2x128 form plus440 padding bytes. This does not load all5800 helper banks or the full model.',
             topology_scope='One internal collector and one internal header with child inputs, fixed color parity. Leaf/parity variants, native producer bindings, device epoch control and actual successor remain unqualified.',
             artifact_reused_from=reuse['source_attempt'],artifact_sha256=artifact['sha256'],
             preserved_host_failure='mlp-chunk-hw-001: top-level run driver executed on import before physical backend selection. No runtime job was submitted; original failed cleanup receipt retained, separate account audit proved no owned allocation. Callable main/import guard fixed002.',
             driver_adaptation=adaptation,hardware_jobs_added=2,jobs=jobs,all_owned_jobs_released=True,
             lifecycle_seconds=dict(compile=compile_audit['wall_seconds'],failed_host_entry=failed_host['wall_seconds'],run=run_audit['wall_seconds']),
             lifecycle_note='Host scheduling/startup/validation/cleanup durations, not inference latency, device cycles or provider billed node-hours.',
             complete_subgraph_executable=False,compiled_all_role_sram_admitted=False,full_model=False,
             full_model_speed_target_achieved=False,measured_model_tokens_per_second=None,
             evidence_sha256=inputs)
with a.output.open('x') as f:
    json.dump(summary,f,indent=2);f.write('\n')
print(json.dumps(dict(passed=True,original_tensors=1251,embedding_helper_candidates=5800,
                     physically_qualified_embedding_helpers=2,sram=roles,hardware_jobs_added=2,all_owned_jobs_released=True,
                     complete_subgraph_executable=False,measured_model_tokens_per_second=None)))
