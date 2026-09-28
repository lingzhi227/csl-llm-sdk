"""Recheck original connected numerical captures after physical jobs release."""
import argparse,json,re,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('attempt');args=parser.parse_args()
    if not re.fullmatch('[0-9]{3}',args.attempt):raise ValueError('Attempt')
    name='gdn-fusion-hw-'+args.attempt;out=ROOT/'evidence'/name/'numerical-audit.json'
    if out.exists():raise ValueError('Frozen audit')
    script='root_name='+repr('/srv/qwen38-singlewse-hardware/'+name)+'\n'+'''import os,resource,sys
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
resource.setrlimit(resource.RLIMIT_AS,(768<<20,768<<20));resource.setrlimit(resource.RLIMIT_CPU,(30,30))
import json,numpy as np
from pathlib import Path
root=Path(root_name);os.chdir(root);sys.path.insert(0,str(root))
from source_gate import verify as source_verify
from backend import file_sha256
from reference.gdn_fusion_audit import verify
from run import timing_record
source_verify();read=lambda n:json.loads((root/n).read_text())
complete=read('COMPLETE.json');result=read('result.json');meta=read('fixture.json')
assert complete['all_owned_jobs_released'] and result['passed'] and result['physical'] and result['normal_stop']
assert result['groups']==list(range(16)) and result['workers']==753 and result['full_original_value_coverage'] and len(result['records'])==8
assert result['positions']==6 and result['reset_replay_positions']==2 and not result['host_injected_recurrent_results']
jobs=[]
for name in ['compile-audit.json','run-audit.json']:
 a=read(name);assert a['exit_code']==0 and not a['error'] and not a['cleanup_errors']
 assert a['jobs'] and all(j['phase']=='SUCCEEDED' and j['released'] for j in a['jobs']);jobs.extend(j['id'] for j in a['jobs'])
assert file_sha256(root/'fixture.npz')==meta['fixture_sha256']==result['fixture_sha256']
with np.load(root/'actual.npz',allow_pickle=False) as actual,np.load(root/'fixture.npz',allow_pickle=False) as expected:
 report=verify(actual,expected)
 for r in result['records']:
  assert all(r[k] for k in ['frontend_interval','recurrent_state_interval','core_interval','gated_interval','exact_history','all_return_markers','delayed_source_callback'])
  suffix=str(int(r['replay']))+'_'+str(r['position'])
  ft=actual['frontend_ticks_'+suffix];wt=actual['worker_ticks_'+suffix]
  assert ft.shape==(16,21) and wt.shape==(753,24)
  assert timing_record(ft,wt)==r['timing']
report.update(physical=True,all_jobs_succeeded_and_released=jobs,
 hashes={n:file_sha256(root/n) for n in ['source-manifest.json','fixture.npz','actual.npz','result.json','compile-audit.json','run-audit.json']},
 scope='All16 original frontend groups directly connected to753 recurrent slices/48 heads, with actual gated outputs. Original projected inputs are a diagnostic boundary; no output projection, full layer or model speed admission.')
with (root/'numerical-audit.json').open('x') as f:json.dump(report,f,indent=2);f.write('\\n')
print(json.dumps(report))
'''
    command='/opt/cerebras/venv/bin/python -c '+shlex.quote(script)
    r=subprocess.run(['sh','/path/to/alcf-session.sh','host',command],capture_output=True,text=True,check=True,timeout=45)
    report=json.loads(r.stdout)
    with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report))


if __name__=='__main__':main()
