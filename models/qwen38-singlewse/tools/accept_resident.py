"""Accept only released physical sentence runs with the frozen full CPU oracle."""
import argparse,hashlib,json,re,shlex,statistics,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--physical',required=True);p.add_argument('--reference',required=True);a=p.parse_args()
assert re.fullmatch('resident-generation-hw-[0-9]{3}',a.physical) and re.fullmatch('reference-sequence-[0-9]{3}',a.reference)
def fetch(prefix,root,names):
    code='root='+repr(root)+'\nnames='+repr(names)+'\n'+'''
import json
from pathlib import Path
p=Path(root);files={n:(p/n).read_text() for n in names};assert sum(len(v.encode()) for v in files.values())<2<<20
print(json.dumps(files))
'''
    r=subprocess.run(prefix+['python3 -c '+shlex.quote(code)],capture_output=True,text=True,check=True,timeout=30)
    return json.loads(r.stdout)
physical=fetch(['sh','/path/to/alcf-session.sh','host'],'/srv/qwen38-singlewse-hardware/'+a.physical,
    ['COMPLETE.json','CANDIDATE.json','initialization.json','source-manifest.json','artifact.json','sram.json','binding.json','reuse-artifact.json',
     *[f'request-{i:02d}/generation.json' for i in range(4)],*[f'request-{i:02d}/trace-check.json' for i in range(4)]])
reference=fetch(['ssh','workstation'],'/srv/model-storage/qwen38-singlewse/runs/'+a.reference,
    ['COMPLETE.json','qualification.json','source-manifest.json','candidate-origin.json','candidate/generation.json'])
P={n:json.loads(v) for n,v in physical.items()};R={n:json.loads(v) for n,v in reference.items()}
criteria=json.loads((ROOT/'configs/full-acceptance-v1.json').read_text());frozen=json.loads((ROOT/'evidence/full-acceptance-frozen.json').read_text())
assert hashlib.sha256((ROOT/'configs/full-acceptance-v1.json').read_bytes()).hexdigest()==frozen['sha256']
assert P['COMPLETE.json']['all_owned_jobs_released'] and P['COMPLETE.json']['physical_execution']
jobs=[j for s in P['COMPLETE.json']['stages'] for j in s['jobs']]
assert len(jobs)==1 and all(j['released'] and not j['cancelled'] and j['phase']=='SUCCEEDED' for j in jobs)
assert P['CANDIDATE.json']['candidate_execution_complete'] and P['CANDIDATE.json']['normal_stop'] and P['CANDIDATE.json']['physical']
assert P['CANDIDATE.json']['acceptance_sha256']==frozen['sha256']==R['COMPLETE.json']['acceptance_sha256']
assert P['sram.json']['passed'] and P['sram.json']['application_pes']==870000
assert P['binding.json']['passed'] and P['binding.json']['application_pes']==870000
assert P['artifact.json']['sha256']==P['sram.json']['artifact_sha256']==P['reuse-artifact.json']['artifact_sha256']
assert P['binding.json']['artifact_sha256']==P['artifact.json']['sha256']
init=P['initialization.json'];assert init['passed'] and init['text_tensors_loaded']==1251 and init['original_weight_pes_loaded']==849313
assert init['role_sha256']==P['binding.json']['role_sha256'] and init['config_sha256']==P['binding.json']['config_sha256']
assert not init['per_token_weight_upload']
assert R['COMPLETE.json']['passed'] and not R['COMPLETE.json']['physical']
assert R['COMPLETE.json'].get('violation_count',0)==0
assert R['candidate-origin.json']['remote']=='/srv/qwen38-singlewse-hardware/'+a.physical
assert physical['request-00/generation.json']==reference['candidate/generation.json']
assert hashlib.sha256(physical['request-00/generation.json'].encode()).hexdigest()==R['qualification.json']['candidate_generation_sha256']
assert R['source-manifest.json']['files']['run.py']==hashlib.sha256((ROOT/'reference/sequence/run.py').read_bytes()).hexdigest()
assert R['source-manifest.json']['files']['full-acceptance-v1.json']==P['source-manifest.json']['files']['full-acceptance-v1.json']==frozen['sha256']
first=P['request-00/generation.json'];positions=first['processed_positions']
assert R['COMPLETE.json']['comparison_checks']==R['COMPLETE.json']['expected_comparison_checks']==positions*66
assert len(R['COMPLETE.json']['token_checks'])==len(first['generated_ids'])
request_calls=0;measured=[]
for index,spec in enumerate(criteria['run_order']):
    run=P[f'request-{index:02d}/generation.json'];trace=P[f'request-{index:02d}/trace-check.json']
    workload=next(w for w in criteria['workloads'] if w['name']==spec['workload'])
    assert run==P['CANDIDATE.json']['runs'][index]
    assert run['physical'] and run['capture_enabled']==spec['capture'] and run['prompt_ids']==workload['input_ids']
    assert run['all_layers']==64 and run['vocabulary']==248320 and run['no_hidden_or_next_token_upload']
    assert run['eos_reached'] and run['generated_ids'][-1]==workload['eos_id']
    assert run['processed_positions']==len(run['prompt_ids'])+len(run['generated_ids'])-1<=96
    request_calls+=len(run['generated_ids']);assert trace['all870000_endpoints_checked'] and trace['request_calls']==request_calls
    if spec['workload']=='sunny_morning':assert run['generated_ids']==first['generated_ids']
    if spec['workload']=='sunny_morning' and not spec['capture']:
        ttft=run['ttft_seconds'];elapsed=run['end_to_end_seconds'];assert 0<ttft<elapsed
        measured.append(dict(request=index,ttft_seconds=ttft,end_to_end_seconds=elapsed,generated_tokens_including_eos=len(run['generated_ids']),
            dependent_decode_tokens_per_second_including_eos=(len(run['generated_ids'])-1)/(elapsed-ttft),
            dependent_step_seconds=[s['dependent_step_seconds'] for s in run['steps'][1:]]))
assert len(measured)==2
result=dict(passed=True,full_model_complete=True,physical=True,model=criteria['model'],revision=criteria['revision'],
    system_count=1,architecture='WSE-3',context_capacity=96,all_layers=64,vocabulary=248320,text_tensors=1251,
    generated_text=R['COMPLETE.json']['decoded_text'],generated_ids=first['generated_ids'],
    numerical_comparisons=R['COMPLETE.json']['comparison_checks'],physical_attempt=a.physical,reference_attempt=a.reference,
    artifact_sha256=P['artifact.json']['sha256'],acceptance_sha256=frozen['sha256'],physical_jobs=jobs,
    initialization_seconds=P['CANDIDATE.json']['runtime_initialization_seconds'],weight_initialization_seconds=init['seconds'],
    measured_capture_disabled_requests=measured,scope='Full text inference, excluding unused vision/MTP branches; actual unoptimized resident implementation measurements')
out=ROOT/'evidence'/a.physical/'acceptance';out.mkdir()
for scope,files in [('physical',physical),('reference',reference)]:
    folder=out/scope;folder.mkdir()
    for name,raw in files.items():
        target=folder/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(raw)
(out/'COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
