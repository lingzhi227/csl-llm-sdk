"""Collect compact progress and aggregate errors without transferring raw arrays."""
import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('attempt')
args = parser.parse_args()
assert re.fullmatch('reference-sequence-[0-9]{3}', args.attempt)
remote = '/srv/model-storage/qwen38-singlewse/runs/' + args.attempt
code = 'root=' + repr(remote) + '\n' + '''
import json,math,hashlib
from pathlib import Path
p=Path(root);files={}
for name in ['progress.json','qualification.json','COMPLETE.json','FAILURE.json']:
 f=p/name
 if f.exists():
  assert f.stat().st_size<1<<20
  files[name]=json.loads(f.read_text())
summary={};records=0;first_failure=None
f=p/'comparisons.jsonl'
if f.exists():
 with f.open() as stream:
  for line in stream:
   assert len(line)<16384
   try:r=json.loads(line)
   except json.JSONDecodeError:
    assert not line.endswith('\\n');break
   records+=1;kind=r['kind'];s=summary.setdefault(kind,dict(checks=0,all_passed=True,min_cosine=1.0))
   s['checks']+=1;s['all_passed'] &= r['passed'];s['min_cosine']=min(s['min_cosine'],r['cosine'])
   for key in ['relative_l2','max_abs_error','rms_abs_error','max_abs_error_over_reference_rms']:
    assert math.isfinite(r[key]);s['max_'+key]=max(s.get('max_'+key,0),r[key])
   if not r['passed'] and first_failure is None:first_failure=r
print(json.dumps(dict(files=files,numerical=dict(records=records,by_kind=summary,first_failure=first_failure))))
'''
process = subprocess.run(['ssh','workstation','/usr/bin/python3 -c '+shlex.quote(code)],
    capture_output=True,text=True,check=True,timeout=30)
result=json.loads(process.stdout)
folder=ROOT/'evidence'/args.attempt
folder.mkdir(exist_ok=True)
(folder/'latest-reference.json').write_text(json.dumps(result,indent=2)+'\n')
for name in ['COMPLETE.json','FAILURE.json','qualification.json']:
    if name in result['files']:
        target=folder/name
        raw=json.dumps(result['files'][name],indent=2)+'\n'
        if target.exists():assert target.read_text()==raw
        else:target.write_text(raw)
if any(name in result['files'] for name in ['COMPLETE.json','FAILURE.json']):
    target=folder/'comparison-summary.json'
    raw=json.dumps(result['numerical'],indent=2)+'\n'
    if target.exists():assert target.read_text()==raw
    else:target.write_text(raw)
print(json.dumps(dict(progress=result['files'].get('progress.json'),numerical=result['numerical'],
    complete='COMPLETE.json' in result['files'],failed='FAILURE.json' in result['files'])))
