"""Bounded fresh physical-capture audit, including bitwise FP32 reset replay."""
import argparse,json,re,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
 parser=argparse.ArgumentParser();parser.add_argument('attempt');args=parser.parse_args()
 if not re.fullmatch('[0-9]{3}',args.attempt):raise ValueError('Attempt')
 name='gdn-columns-hw-'+args.attempt;out=ROOT/'evidence'/name/'numerical-audit.json'
 if out.exists():raise ValueError('Frozen audit')
 script='root_name='+repr('/srv/qwen38-singlewse-hardware/'+name)+'\n'+'''import os,resource
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
resource.setrlimit(resource.RLIMIT_AS,(768<<20,768<<20));resource.setrlimit(resource.RLIMIT_CPU,(30,30))
import hashlib,json
from pathlib import Path
import numpy as np
root=Path(root_name);read=lambda n:json.loads((root/n).read_text())
complete=read('COMPLETE.json');result=read('result.json');meta=read('fixture.json')
assert complete['all_owned_jobs_released'] and result['passed'] and result['physical'] and result['normal_stop']
assert result['heads']==48 and result['workers']==753 and result['full_original_value_coverage'] and len(result['records'])==8
jobs=[]
for name in ['compile-audit.json','run-audit.json']:
 a=read(name);assert a['exit_code']==0 and not a['error'] and not a['cleanup_errors']
 assert a['jobs'] and all(j['phase']=='SUCCEEDED' and j['released'] for j in a['jobs']);jobs.extend(j['id'] for j in a['jobs'])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
assert sha(root/'fixture.npz')==meta['fixture_sha256']==result['fixture_sha256']
actual=np.load(root/'actual.npz',allow_pickle=False);expected=np.load(root/'fixture.npz',allow_pickle=False)
maximum_ratio=0.;maximum_relative_l2=0.;states=0;outputs=0
for replay,length in [(0,6),(1,2)]:
 for position in range(length):
  state=actual[f'state_{replay}_{position}'];out=actual[f'output_{replay}_{position}']
  assert state.shape==(48,128,128) and state.dtype==np.float32 and out.shape==(48,128) and out.dtype==np.uint16
  reference=expected[f'state_{position}'];bound=expected[f'bound_{position}'];diff=np.abs(state.astype(np.float64)-reference)
  assert np.isfinite(state).all() and np.all(diff<=bound)
  values=(out.astype(np.uint32)<<16).view(np.float32).astype(np.float64)
  assert np.isfinite(values).all() and np.all(values>=expected[f'lower_{position}']) and np.all(values<=expected[f'upper_{position}'])
  maximum_ratio=max(maximum_ratio,float(np.max(diff/np.maximum(bound,np.finfo(np.float64).tiny))))
  maximum_relative_l2=max(maximum_relative_l2,float(np.linalg.norm(diff)/max(np.linalg.norm(reference),1e-30)))
  if replay:
   assert np.array_equal(state.view(np.uint32),actual[f'state_0_{position}'].view(np.uint32))
   assert np.array_equal(out,actual[f'output_0_{position}'])
  states+=state.size;outputs+=out.size
report=dict(passed=True,physical=True,state_elements_verified=states,bf16_outputs_verified=outputs,paired_frames=outputs//2,bitwise_reset_replay=True,
 maximum_state_error_to_bound_ratio=maximum_ratio,maximum_state_relative_l2=maximum_relative_l2,all_jobs_succeeded_and_released=jobs,
 hashes={n:sha(root/n) for n in ['source-manifest.json','fixture.npz','actual.npz','result.json','compile-audit.json','run-audit.json']},
 scope='All48 heads/753 original value slices over6continuous+2reset positions; exact-port component with injected original-reference frontend inputs, not a complete layer/model or speed result')
with (root/'numerical-audit.json').open('x') as f:json.dump(report,f,indent=2);f.write('\\n')
print(json.dumps(report))
'''
 command='/opt/cerebras/venv/bin/python -c '+shlex.quote(script)
 r=subprocess.run(['sh','/path/to/alcf-session.sh','host',command],capture_output=True,text=True,check=True,timeout=45)
 report=json.loads(r.stdout)
 with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(report))

if __name__=='__main__':main()
