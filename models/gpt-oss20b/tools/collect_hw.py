"""Collect compact completed hardware receipts; retain binaries on remote storage."""
import argparse,json,re,shlex,subprocess
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('name')
p.add_argument('--capture-only',action='store_true',help='Collect a released full-model capture without claiming numerical acceptance')
args=p.parse_args()
if not re.fullmatch(r'[a-z][a-z0-9-]+-hw-[0-9]{3}',args.name):raise ValueError('Invalid experiment name')
root='/srv/gpt-oss20b-hardware/'+args.name
names=['COMPLETE.json','result.json','sram.json','artifact.json','compile-audit.json','run-audit.json',
       'observations.json','fixture.json','math-fixture.json','transformer-fixture.json','expert-fixture.json','router-fixture.json',
       'moe-fixture.json','attention-fixture.json','full-fixture.json','weight-load.json','weight-retention.json',
       'admission.json','PRE_RUN_ADMISSION.json','reused-compile.json','compile.jobs','run.jobs','source-manifest.json',
       'norm-fixture.json','PHYSICAL_CAPTURE_COMPLETE.json','NUMERICAL_REVIEW_REQUIRED.json','QUALIFICATION.json']
receipt='PHYSICAL_CAPTURE_COMPLETE.json' if args.capture_only else 'COMPLETE.json'
script='root_name='+repr(root)+'\nnames='+repr(names)+'\nreceipt='+repr(receipt)+'\n'+'''
from pathlib import Path
import json
root=Path(root_name)
assert json.loads((root/receipt).read_text())['all_owned_jobs_released']
files={n:(root/n).read_text() for n in names if (root/n).exists()}
assert sum(len(v.encode()) for v in files.values())<2097152
print(json.dumps(files))
'''
r=subprocess.run(['sh','/path/to/alcf-session.sh','host','python3 -c '+shlex.quote(script)],
                  capture_output=True,text=True,check=True,timeout=30)
out=Path(__file__).resolve().parents[1]/'evidence'/args.name
out.mkdir(parents=True,exist_ok=True)
for name,raw in json.loads(r.stdout).items():(out/name).write_text(raw)
print((out/receipt).read_text())
