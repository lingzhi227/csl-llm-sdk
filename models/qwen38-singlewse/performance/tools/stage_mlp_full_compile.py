"""Bounded workstation whole-wafer compiler admission; no physical job."""
import argparse,ast,hashlib,io,json,re,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--candidate',required=True);p.add_argument('--profiles',required=True);p.add_argument('--seconds',type=int,choices=[600,1800],default=600);a=p.parse_args()
assert re.fullmatch('[0-9]{3}',a.attempt) and re.fullmatch('mlp-executable-candidate-[0-9]{3}',a.candidate)
assert re.fullmatch('mlp-integrated-compile-[0-9]{3}',a.profiles)
name='mlp-full-compile-'+a.attempt;out=ROOT/'performance/evidence'/name
if out.exists():raise ValueError('Frozen attempt')
candidate=ROOT/'performance/evidence'/a.candidate;meta=json.loads((candidate/'candidate.json').read_text())
profiles=ROOT/'performance/evidence'/a.profiles
assert json.loads((profiles/'COMPLETE.json').read_text())['sram_passed']
prior=json.loads((profiles/'source-manifest.json').read_text())['files']
files={}
for n,h in meta['source_sha256'].items():
 b=(candidate/n).read_bytes();assert hashlib.sha256(b).hexdigest()==h
 if n not in ['layout.csl','mlp_idle.csl']:assert prior[n]==h,n
 files[n]=b
files['candidate.json']=(candidate/'candidate.json').read_bytes()
for n,path in {'source_gate.py':'runtime/source_gate.py','elf_inventory.py':'runtime/elf_inventory.py','check_sram.py':'performance/runtime/check_sram.py','placement.py':'performance/runtime/placement.py'}.items():files[n]=(ROOT/path).read_bytes()
active=subprocess.run(['ssh','workstation','systemctl --user list-units --type=service --state=active,activating,deactivating --no-legend qwen38-single-*'],capture_output=True,text=True,check=True,timeout=20)
if active.stdout.strip():raise ValueError('Existing workstation owner: '+active.stdout)
files['execute.py']=br'''import json,os,signal,subprocess,time
from pathlib import Path
from source_gate import verify
from check_sram import check
verify();started=time.monotonic()
try:
 with Path('compile.log').open('x') as log:
  proc=subprocess.Popen(['/opt/cerebras/sdk/2.10.1/cslc','layout.csl','--arch=wse3','--fabric-dims=762,1172','--fabric-offsets=4,1','--memcpy','--channels=1','--max-parallelism=1','-o','out'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   code=proc.wait(timeout=600)
   if code:raise RuntimeError('compile exit '+str(code))
  finally:
   if proc.poll() is None:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=2)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=2)
 result=check(Path.cwd(),application=(750,1160),fabric=(762,1172))
 verify();Path('COMPLETE.json').write_text(json.dumps(dict(passed=True,physical=False,executed=False,application=[750,1160],sram_passed=result['passed'],seconds=time.monotonic()-started))+'\n')
except BaseException as e:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(e).__name__,message=str(e)))+'\n');raise
'''
files['execute.py']=files['execute.py'].replace(b'timeout=600',('timeout='+str(a.seconds)).encode())
for n,b in files.items():
 if n.endswith('.py'):ast.parse(b)
files['source-manifest.json']=(json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2)+'\n').encode()
remote='/srv/model-storage/qwen38-singlewse/runs/'+name
payload=io.BytesIO()
with tarfile.open(fileobj=payload,mode='w') as tar:
 for n,b in files.items():
  member=tarfile.TarInfo(n);member.size=len(b);tar.addfile(member,io.BytesIO(b))
subprocess.run(['ssh','workstation','mkdir '+shlex.quote(remote)+' && tar -xf - -C '+shlex.quote(remote)],input=payload.getvalue(),check=True,timeout=30)
unit='qwen38-single-'+name
cmd=['systemd-run','--user','--unit='+unit,'--property=MemoryMax=4G','--property=MemorySwapMax=0','--property=TasksMax=64','--property=CPUQuota=200%','--property=AllowedCPUs=6,7','--property=RuntimeMaxSec='+str(a.seconds+50),'--property=TimeoutStopSec=5','--property=KillMode=control-group','--property=LimitFSIZE=536870912','--property=LimitCORE=0','--working-directory='+remote,'--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=PYTHONDONTWRITEBYTECODE=1','/usr/bin/taskset','--cpu-list','6,7','/usr/bin/flock','-n','/srv/cerebras-workstation/heavy.lock','/usr/bin/python3','execute.py']
subprocess.run(['ssh','workstation',shlex.join(cmd)],check=True,timeout=30)
out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
(out/'dispatch.json').write_text(json.dumps(dict(remote=remote,unit=unit,physical=False,memory_max=4<<30,swap_max=0,compiler_seconds=a.seconds,total_seconds=a.seconds+50),indent=2)+'\n')
print(json.dumps(dict(remote=remote,unit=unit,physical=False)))
