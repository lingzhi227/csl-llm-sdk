"""Bounded compact2D projection qualification from shared flow/epoch lowering."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'performance'))
from spatial.projection_flow import ProjectionFlow,ProjectionSpec
from spatial.sdk_leases import audit_explicit_sdk_leases
from tools.audit_projection_flow import audit
p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--k-axis',choices=['x','y'],default='x');a=p.parse_args()
if len(a.attempt)!=3 or not a.attempt.isdigit():raise ValueError('Three-digit attempt required')
name='compact-projection-sim-'+a.attempt;remote='/srv/model-storage/qwen38-singlewse/runs/'+name
out=ROOT/'performance/evidence'/name
if out.exists():raise ValueError('Frozen local attempt exists')
active=subprocess.run(['ssh','workstation','systemctl --user list-units --type=service --state=active,activating,deactivating --no-legend qwen38-single-*'],capture_output=True,text=True,check=True,timeout=20)
if active.stdout.strip():raise RuntimeError('Existing workstation owner: '+active.stdout)
ref=ROOT/'performance/evidence/full-bf16-matrix-sim-017';original=json.loads((ref/'fixture.json').read_text())
assert json.loads((ref/'COMPLETE.json').read_text())['result']['passed'] and json.loads((ref/'workstation-release.json').read_text())['workstation_released']
flow=ProjectionFlow(ProjectionSpec(k_axis=a.k_axis));document=flow.document()
for old,new in zip(original['workers'],document['workers']):
 for key in ['fp8_slots','bf16_slots','target_slot']:new[key]=old[key]
 new['global_xy']=[new['xy'][0]+119,new['xy'][1]+705]
 new['reference_global_xy']=old['global_xy'];new['reference_bank_rank']=old['bank_rank'];new['reference_eligible_rank']=old['eligible_rank']
document['global_origin']=[119,705];document['full_matrix_shape']=[48,5120];document['full_model']=False
files={n:(ROOT/'performance/probes/compact_projection'/n).read_bytes() for n in ['run.py','prepare.py']}
files['matrix_epoch.csl']=(ROOT/'performance/protocols/matrix_epoch.csl').read_bytes()
for n in ['fp8_shape.csl','bf16_shape.csl']:files[n]=(ROOT/'performance/probes/native_shapes'/n).read_bytes()
files['fp8_unpack_shift.csl']=(ROOT/'performance/csl/fp8_unpack_shift.csl').read_bytes()
files['flow-audit.json']=(json.dumps(audit(document),indent=2)+'\n').encode()
files['resource-audit.json']=(json.dumps(audit_explicit_sdk_leases({n:b for n,b in files.items() if n.endswith('.csl')}),indent=2)+'\n').encode()
files['reference-fixture.json']=(ref/'fixture.json').read_bytes()
for n in ['backend.py','source_gate.py','elf_inventory.py']:files[n]=(ROOT/'runtime'/n).read_bytes()
for n in ['check_sram.py','placement.py']:files[n]=(ROOT/'performance/runtime'/n).read_bytes()
files['layout.csl']=flow.emit_layout(original['workers']).encode();files['region.json']=(json.dumps(document,separators=(',',':'))+'\n').encode()
w,h=flow.width,flow.height
files['execute.py']=(f'application=({w},{h});fabric=({w+7},{h+2})\n'+'''import subprocess,json,time,os,signal
from pathlib import Path
from source_gate import verify
from check_sram import check
verify();started=time.monotonic();sdk='/opt/cerebras/sdk/2.10.1/'
commands=[['prepare',['/usr/bin/python3','prepare.py']],['compile',[sdk+'cslc','layout.csl','--arch=wse3','--fabric-dims=%d,%d'%fabric,'--fabric-offsets=4,1','--memcpy','--channels=2','--max-parallelism=1','-o','out']],['run',[sdk+'cs_python','run.py']]]
try:
 for phase,cmd in commands:
  with Path(phase+'.log').open('x') as out:
   budget=900 if phase=='run' else 240
   proc=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT,start_new_session=True);deadline=time.monotonic()+budget
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
  if phase=='compile':assert check(Path.cwd(),application=application,fabric=fabric)['application_pes']==application[0]*application[1]
 verify();Path('COMPLETE.json').write_text(json.dumps(dict(physical=False,seconds=time.monotonic()-started,result=json.loads(Path('result.json').read_text())))+'\\n')
except BaseException as e:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(e).__name__,message=str(e)))+'\\n');raise
''').encode()
files['source-manifest.json']=(json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2)+'\n').encode()
assert sum(map(len,files.values()))<8<<20
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w') as tar:
 for n,b in files.items():
  item=tarfile.TarInfo(n);item.size=len(b);tar.addfile(item,io.BytesIO(b))
subprocess.run(['ssh','workstation','mkdir '+shlex.quote(remote)+' && tar -xf - -C '+shlex.quote(remote)],input=buffer.getvalue(),check=True,timeout=30)
subprocess.run(['ssh','workstation','cp --reflink=auto /srv/model-storage/qwen38-singlewse/runs/full-bf16-matrix-sim-017/fixture.npz '+shlex.quote(remote+'/fixture.npz')],check=True,timeout=30)
unit='qwen38-single-'+name
command=['systemd-run','--user','--unit='+unit,'--property=MemoryMax=2G','--property=MemorySwapMax=0','--property=TasksMax=64','--property=CPUQuota=200%','--property=AllowedCPUs=6,7','--property=RuntimeMaxSec=1420','--property=TimeoutStopSec=5','--property=KillMode=control-group','--property=LimitFSIZE=536870912','--property=LimitCORE=0','--working-directory='+remote,'--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=PYTHONDONTWRITEBYTECODE=1','/usr/bin/taskset','--cpu-list','6,7','/usr/bin/flock','-n','/srv/cerebras-workstation/heavy.lock','/usr/bin/python3','execute.py']
subprocess.run(['ssh','workstation',shlex.join(command)],check=True,timeout=30)
out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json']);(out/'dispatch.json').write_text(json.dumps(dict(remote=remote,unit=unit,physical=False,memory_max=2<<30,swap_max=0,runtime_max_seconds=1420,prepare_compile_limit_seconds=240,run_limit_seconds=900,memcpy_channels=2),indent=2)+'\n')
print(json.dumps(dict(remote=remote,unit=unit,application=[w,h],static_max_link_words=document['cost']['max_link_words_including_teardown'])))
