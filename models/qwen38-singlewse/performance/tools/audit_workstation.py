"""Record an invocation's terminal unit, shared-lock release and compact log diagnostics."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(); parser.add_argument('attempt'); args = parser.parse_args()
if not re.fullmatch('(?:[a-z][a-z0-9-]*-sim|projection-audit|atlas-audit)-[0-9]{3}', args.attempt):
    raise ValueError('Simulator or metadata-audit attempt required')
destination = ROOT / 'evidence' / args.attempt / 'workstation-release.json'
if destination.exists():
    raise ValueError('Frozen release already exists')
script = 'name=' + repr(args.attempt) + '\n' + '''import datetime,fcntl,hashlib,json,subprocess
from pathlib import Path
unit='qwen38-single-'+name+'.service'
result=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','-p','SubState','-p','Result','-p','ExecMainStatus'],capture_output=True,text=True,check=True,timeout=10).stdout
status=dict(line.split('=',1) for line in result.splitlines() if '=' in line)
with open('/srv/cerebras-workstation/heavy.lock','a') as f:
 try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);free=True
 except BlockingIOError:free=False
active=subprocess.run(['systemctl','--user','list-units','--type=service','--state=running','--no-legend','qwen38-single-*'],capture_output=True,text=True,check=True,timeout=10).stdout.strip()
root=Path('/srv/model-storage/qwen38-singlewse/runs')/name
logs={}
for name in ['prepare.log','compile.log','run.log','sim.log']:
 p=root/name
 if p.is_file():
  size=p.stat().st_size;assert size<=33554432
  digest=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1<<20),b''):digest.update(b)
   f.seek(max(0,size-2048));tail=f.read().decode(errors='replace')
  logs[name]=dict(bytes=size,sha256=digest.hexdigest(),tail=tail)
released=status.get('ActiveState') in ['inactive','failed'] and free and not active
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),unit=unit,status=status,shared_lock_available=free,own_running_services=active,workstation_released=released,physical=False,logs=logs)))
'''
result = subprocess.run(['ssh', 'workstation', 'python3 -c ' + shlex.quote(script)], capture_output=True, text=True, check=True, timeout=30)
audit = json.loads(result.stdout)
if not audit['workstation_released']:
    raise RuntimeError('Workstation resources still active: ' + result.stdout)
with destination.open('x') as f:
    json.dump(audit, f, indent=2); f.write('\n')
print(json.dumps({k: v for k, v in audit.items() if k != 'logs'}))
