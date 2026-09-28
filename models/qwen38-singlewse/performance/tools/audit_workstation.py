"""Record an invocation's terminal unit, shared-lock release and compact log diagnostics."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(); parser.add_argument('attempt'); args = parser.parse_args()
if not re.fullmatch('(?:[a-z][a-z0-9-]*-sim|projection-audit|atlas-audit|columnar-audit|coupled-audit|retiled-comparison-audit|layer-backend-compile|layer-projection-compile|layer-weight-audit)-[0-9]{3}', args.attempt):
    raise ValueError('Simulator or metadata-audit attempt required')
destination = ROOT / 'evidence' / args.attempt / 'workstation-release.json'
if destination.exists():
    raise ValueError('Frozen release already exists')
unit='qwen38-single-'+args.attempt+'.service'
remote='/srv/model-storage/qwen38-singlewse/runs/'+args.attempt
dispatch_hash=None
if args.attempt.startswith('layer-'):
    import hashlib
    dispatch=destination.parent/'dispatch.json';raw=dispatch.read_bytes();record=json.loads(raw)
    if record['remote']!=remote or record['physical'] is not False:
        raise ValueError('Invocation identity differs from the bounded workstation dispatch')
    unit=record['unit'].removesuffix('.service')+'.service'
    if not re.fullmatch(r'qwen38-single-[a-z0-9_-]+\.service',unit):raise ValueError('Unit identity')
    dispatch_hash=hashlib.sha256(raw).hexdigest()
script = 'name=' + repr(args.attempt) + '\nunit='+repr(unit)+'\nexpected_remote='+repr(remote)+'\ndispatch_hash='+repr(dispatch_hash)+'\n' + '''import datetime,fcntl,hashlib,json,os,subprocess
from pathlib import Path
result=subprocess.run(['systemctl','--user','show',unit,'-p','LoadState','-p','ActiveState','-p','SubState','-p','Result','-p','ExecMainStatus','-p','MainPID','-p','ControlGroup','-p','WorkingDirectory'],capture_output=True,text=True,check=True,timeout=10).stdout
status=dict(line.split('=',1) for line in result.splitlines() if '=' in line)
journal=[]
if dispatch_hash and status.get('WorkingDirectory')!=expected_remote:
 if status.get('LoadState')!='not-found':raise ValueError('Unit working directory differs from frozen dispatch')
 # systemd garbage-collects successful transient units. Do not invent an exit
 # code for an unloaded unit: retain the invocation journal and independently
 # prove there is no remaining process with this run's cwd or unit membership.
 raw=subprocess.run(['journalctl','--user-unit='+unit,'--no-pager','-n','32','-o','json'],capture_output=True,text=True,check=True,timeout=10).stdout
 for line in raw.splitlines():
  item=json.loads(line)
  if item.get('USER_UNIT',item.get('_SYSTEMD_USER_UNIT'))==unit:
   journal.append({k:item[k] for k in ['USER_UNIT','USER_INVOCATION_ID','MESSAGE','__REALTIME_TIMESTAMP','CPU_USAGE_NSEC'] if k in item})
 if not journal or not any('CPU_USAGE_NSEC' in j for j in journal):raise ValueError('No invocation resource-end journal for unloaded unit')
matching_processes=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  if proc.stat().st_uid!=os.getuid():continue
  cwd=str((proc/'cwd').resolve(strict=True));groups=(proc/'cgroup').read_text()
  if cwd==expected_remote or cwd.startswith(expected_remote+'/') or '/'+unit in groups:matching_processes.append(int(proc.name))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
cgroup=status.get('ControlGroup','');members=[]
if cgroup:
 path=Path('/sys/fs/cgroup')/cgroup.lstrip('/')
 if path.exists():
  for member in path.rglob('cgroup.procs'):members.extend(member.read_text().split())
with open('/srv/cerebras-workstation/heavy.lock','a') as f:
 try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);free=True
 except BlockingIOError:free=False
active=subprocess.run(['systemctl','--user','list-units','--type=service','--state=active,activating,deactivating','--no-legend','qwen38-single-*'],capture_output=True,text=True,check=True,timeout=10).stdout.strip()
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
released=status.get('ActiveState') in ['inactive','failed'] and status.get('MainPID')=='0' and not members and not matching_processes and free and not active
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),unit=unit,status=status,dispatch_sha256=dispatch_hash,invocation_journal=journal,cgroup_members=members,matching_processes=matching_processes,shared_lock_available=free,own_running_services=active,workstation_released=released,physical=False,logs=logs)))
'''
result = subprocess.run(['ssh', 'workstation', 'python3 -c ' + shlex.quote(script)], capture_output=True, text=True, timeout=30)
if result.returncode:raise RuntimeError('Read-only resource audit failed: '+result.stderr[-3000:])
audit = json.loads(result.stdout)
if not audit['workstation_released']:
    raise RuntimeError('Workstation resources still active: ' + result.stdout)
with destination.open('x') as f:
    json.dump(audit, f, indent=2); f.write('\n')
print(json.dumps({k: v for k, v in audit.items() if k != 'logs'}))
