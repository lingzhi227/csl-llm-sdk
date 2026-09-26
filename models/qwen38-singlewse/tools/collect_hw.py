"""Collect compact completed hardware receipts; retain binaries on remote storage."""
import argparse,json,re,shlex,subprocess
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('name');args=p.parse_args()
if not re.fullmatch(r'[a-z][a-z0-9-]+-hw-[0-9]{3}',args.name):raise ValueError('Invalid experiment name')
root='/srv/qwen38-singlewse-hardware/'+args.name
names=['COMPLETE.json','result.json','sram.json','artifact.json','compile-audit.json','run-audit.json',
       'observations.json','fixture.json','math-fixture.json','transformer-fixture.json','expert-fixture.json','router-fixture.json',
       'admission.json','compile.jobs','run.jobs','source-manifest.json']
script='root_name='+repr(root)+'\nnames='+repr(names)+'\n'+'''
from pathlib import Path
import json
root=Path(root_name)
assert json.loads((root/'COMPLETE.json').read_text())['all_owned_jobs_released']
files={n:(root/n).read_text() for n in names if (root/n).exists()}
assert sum(len(v.encode()) for v in files.values())<2097152
print(json.dumps(files))
'''
r=subprocess.run(['sh','/path/to/alcf-session.sh','host','python3 -c '+shlex.quote(script)],
                  capture_output=True,text=True,check=True,timeout=30)
out=Path(__file__).resolve().parents[1]/'evidence'/args.name
out.mkdir(parents=True,exist_ok=True)
for name,raw in json.loads(r.stdout).items():(out/name).write_text(raw)
print((out/'COMPLETE.json').read_text())
