"""Bounded banked regional original-weight dataflow probe; no physical allocation."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import sys

ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--width',type=int,default=2);p.add_argument('--height',type=int,default=3);a=p.parse_args()
sys.path.insert(0,str(ROOT/'performance'));from spatial.banked_region import BankedRegionalPlan
plan=BankedRegionalPlan(width=a.width,height=a.height);plan.validate()
if not (2<=a.width<=8 and 2<=a.height<=8):raise ValueError('This fixture/runtime budget is restricted to at most8x8')
if not (len(a.attempt)==3 and a.attempt.isdigit()):raise ValueError('attempt')
source=ROOT/'performance/probes/regional_bank'
name='regional-bank-sim-'+a.attempt
remote='/srv/model-storage/qwen38-singlewse/runs/'+name
files={n:(source/n).read_bytes() for n in ['pe.csl','run.py','prepare.py']}
files['layout.csl']=plan.emit_layout().encode()
files['region.json']=(json.dumps(plan.document(),indent=2)+'\n').encode()
files['backend.py']=(ROOT/'runtime/backend.py').read_bytes()
files['source_gate.py']=(ROOT/'runtime/source_gate.py').read_bytes()
for n in ['check_sram.py','elf_inventory.py']:files[n]=(ROOT/'runtime'/n).read_bytes()
files['fp8_unpack.csl']=(ROOT/'performance/csl/fp8_unpack.csl').read_bytes()
files['fp8_dot.csl']=(ROOT/'performance/csl/fp8_dot.csl').read_bytes()
files['weights.py']=(ROOT/'runtime/weights.py').read_bytes()
files['tensors.json']=(ROOT/'configs/tensors.json').read_bytes()
w,h=a.width,a.height
files['execute.py']=f'''import subprocess,json,time,os,signal
from pathlib import Path
from source_gate import verify
from check_sram import check
verify();started=time.monotonic();sdk='/opt/cerebras/sdk/2.10.1/'
commands=[['/usr/bin/python3','prepare.py'],[sdk+'cslc','layout.csl','--arch=wse3','--fabric-dims={w+7},{h+2}','--fabric-offsets=4,1','--memcpy','--channels=1','--max-parallelism=1','-o','out'],[sdk+'cs_python','run.py']]
try:
 for phase,cmd in zip(['prepare','compile','run'],commands):
  with Path(phase+'.log').open('x') as out:
   proc=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
   deadline=time.monotonic()+240
   try:
    while proc.poll() is None:
     if time.monotonic()>deadline:raise TimeoutError(phase+' deadline')
     if phase=='run' and Path('sim.log').exists():
      if Path('sim.log').stat().st_size>8388608:raise RuntimeError('sim log budget')
      if 'FATAL:' in Path('sim.log').read_text():raise RuntimeError('Simulator fatal; see frozen sim.log')
     time.sleep(0.2)
    if proc.returncode:raise RuntimeError(phase+' exit '+str(proc.returncode))
   finally:
    if proc.poll() is None:
     os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=2)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=2)
  if phase=='compile':assert check(Path.cwd())['application_pes']=={w*h}
 verify();Path('COMPLETE.json').write_text(json.dumps(dict(physical=False,seconds=time.monotonic()-started,result=json.loads(Path('result.json').read_text())))+'\\n')
except BaseException as e:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(e).__name__,message=str(e)))+'\\n');raise
'''.encode()
manifest={'files':{n:hashlib.sha256(b).hexdigest() for n,b in files.items()}}
files['source-manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w') as tar:
 for n,data in files.items():
  item=tarfile.TarInfo(n);item.size=len(data);tar.addfile(item,io.BytesIO(data))
subprocess.run(['ssh','workstation','mkdir '+shlex.quote(remote)+' && tar -xf - -C '+shlex.quote(remote)],input=b.getvalue(),check=True,timeout=30)
unit='qwen38-single-'+name
cmd=['systemd-run','--user','--unit='+unit,'--property=MemoryMax=2G','--property=MemorySwapMax=0','--property=TasksMax=64','--property=CPUQuota=200%','--property=AllowedCPUs=6,7','--property=RuntimeMaxSec=760','--property=LimitFSIZE=536870912','--property=LimitCORE=0','--working-directory='+remote,'--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','/usr/bin/taskset','--cpu-list','6,7','/usr/bin/flock','-n','/srv/cerebras-workstation/heavy.lock','/usr/bin/python3','execute.py']
subprocess.run(['ssh','workstation',shlex.join(cmd)],check=True,timeout=30)
out=ROOT/'performance/evidence'/name;out.mkdir()
(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
(out/'dispatch.json').write_text(json.dumps(dict(remote=remote,unit=unit,memory_max=2<<30,swap_max=0,runtime_max_seconds=760,physical=False),indent=2)+'\n')
print(json.dumps(dict(remote=remote,unit=unit)))
