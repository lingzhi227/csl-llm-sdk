"""Bounded four-PE equal-MAC shape qualification; no physical allocation."""
import argparse,hashlib,io,json,shlex,subprocess,sys,tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'performance'))
from spatial.native_shapes import document,emit_layout
from spatial.sdk_leases import audit_explicit_sdk_leases

p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--compare-scaling',action='store_true');a=p.parse_args()
if len(a.attempt)!=3 or not a.attempt.isdigit():raise ValueError('Three-digit attempt required')
name='native-shapes-sim-'+a.attempt;remote='/srv/model-storage/qwen38-singlewse/runs/'+name
out=ROOT/'performance/evidence'/name
if out.exists():raise ValueError('Frozen local attempt exists')
active=subprocess.run(['ssh','workstation','systemctl --user list-units --type=service --state=active,activating,deactivating --no-legend qwen38-single-*'],capture_output=True,text=True,check=True,timeout=20)
if active.stdout.strip():raise RuntimeError('Existing workstation owner: '+active.stdout)
files={n:(ROOT/'performance/probes/native_shapes'/n).read_bytes() for n in ['pe.csl','fp8_shape.csl','bf16_shape.csl','prepare.py','run.py']}
files['fp8_unpack_shift.csl']=(ROOT/'performance/csl/fp8_unpack_shift.csl').read_bytes()
files['resource-audit.json']=(json.dumps(audit_explicit_sdk_leases({n:b for n,b in files.items() if n.endswith('.csl')}),indent=2)+'\n').encode()
files['REUSE.json']=(json.dumps(dict(source_modules={n:hashlib.sha256((ROOT/'performance/csl'/n).read_bytes()).hexdigest() for n in ['fp8_dot.csl','bf16_dot.csl']},change='Copied native arithmetic with compile-time row/K extents generalized; established modules remain unchanged.'),indent=2)+'\n').encode()
plan=document(a.compare_scaling);width=plan['application'][0]
files['region.json']=(json.dumps(plan,indent=2)+'\n').encode();files['layout.csl']=emit_layout(a.compare_scaling).encode()
files['banks.json']=(ROOT/'performance/evidence/mixed-bank-sim-002/fixture.json').read_bytes()
files['tensors.json']=(ROOT/'configs/tensors.json').read_bytes();files['weights.py']=(ROOT/'runtime/weights.py').read_bytes()
for n in ['backend.py','source_gate.py','elf_inventory.py']:files[n]=(ROOT/'runtime'/n).read_bytes()
for n in ['check_sram.py','placement.py']:files[n]=(ROOT/'performance/runtime'/n).read_bytes()
files['execute.py']=b'''import subprocess,json,time,os,signal
from pathlib import Path
from source_gate import verify
from check_sram import check
verify();started=time.monotonic();sdk='/opt/cerebras/sdk/2.10.1/'
commands=[['prepare',['/usr/bin/python3','prepare.py']],['compile',[sdk+'cslc','layout.csl','--arch=wse3','--fabric-dims=11,3','--fabric-offsets=4,1','--memcpy','--channels=1','--max-parallelism=1','-o','out']],['run',[sdk+'cs_python','run.py']]]
try:
 for phase,cmd in commands:
  with Path(phase+'.log').open('x') as out:
   proc=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT,start_new_session=True);deadline=time.monotonic()+240
   try:
    while proc.poll() is None:
     if time.monotonic()>deadline:raise TimeoutError(phase+' deadline')
     if phase=='run' and Path('sim.log').exists():
      if Path('sim.log').stat().st_size>8388608:raise RuntimeError('sim log budget')
      if 'FATAL:' in Path('sim.log').read_text():raise RuntimeError('Simulator fatal; frozen sim.log')
     time.sleep(.2)
    if proc.returncode:raise RuntimeError(phase+' exit '+str(proc.returncode))
   finally:
    if proc.poll() is None:
     os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=2)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=2)
  if phase=='compile':assert check(Path.cwd(),application=(4,1),fabric=(11,3))['application_pes']==4
 verify();Path('COMPLETE.json').write_text(json.dumps(dict(physical=False,seconds=time.monotonic()-started,result=json.loads(Path('result.json').read_text())))+'\\n')
except BaseException as e:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(e).__name__,message=str(e)))+'\\n');raise
'''
files['execute.py']=files['execute.py'].replace(b'--fabric-dims=11,3',('--fabric-dims=%d,3'%(width+7)).encode()).replace(b'application=(4,1),fabric=(11,3)',('application=(%d,1),fabric=(%d,3)'%(width,width+7)).encode()).replace(b"['application_pes']==4",("['application_pes']==%d"%width).encode())
files['source-manifest.json']=(json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2)+'\n').encode()
assert sum(map(len,files.values()))<8<<20
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w') as tar:
 for n,b in files.items():
  item=tarfile.TarInfo(n);item.size=len(b);tar.addfile(item,io.BytesIO(b))
subprocess.run(['ssh','workstation','mkdir '+shlex.quote(remote)+' && tar -xf - -C '+shlex.quote(remote)],input=buffer.getvalue(),check=True,timeout=30)
subprocess.run(['ssh','workstation','cp --reflink=auto /srv/model-storage/qwen38-singlewse/runs/mixed-bank-sim-002/fixture.npz '+shlex.quote(remote+'/banks.npz')],check=True,timeout=30)
unit='qwen38-single-'+name
command=['systemd-run','--user','--unit='+unit,'--property=MemoryMax=2G','--property=MemorySwapMax=0','--property=TasksMax=64','--property=CPUQuota=200%','--property=AllowedCPUs=6,7','--property=RuntimeMaxSec=760','--property=TimeoutStopSec=5','--property=KillMode=control-group','--property=LimitFSIZE=536870912','--property=LimitCORE=0','--working-directory='+remote,'--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=PYTHONDONTWRITEBYTECODE=1','/usr/bin/taskset','--cpu-list','6,7','/usr/bin/flock','-n','/srv/cerebras-workstation/heavy.lock','/usr/bin/python3','execute.py']
subprocess.run(['ssh','workstation',shlex.join(command)],check=True,timeout=30)
out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json']);(out/'dispatch.json').write_text(json.dumps(dict(remote=remote,unit=unit,physical=False,memory_max=2<<30,swap_max=0,runtime_max_seconds=760,phase_limit_seconds=240),indent=2)+'\n')
print(json.dumps(dict(remote=remote,unit=unit)))
