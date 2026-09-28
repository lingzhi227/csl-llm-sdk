"""Freeze and run a bounded full-original-bank relocation after SRAM admission."""
import argparse,ast,hashlib,io,json,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
 p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--compile',required=True);a=p.parse_args()
 if len(a.attempt)!=3 or not a.attempt.isdigit() or len(a.compile)!=3 or not a.compile.isdigit():raise ValueError('Attempt')
 name='layer-joint-reference-'+a.attempt;e=ROOT/'performance/evidence';out=e/name;compiled=e/('layer-mlp-compile-'+a.compile)
 if out.exists():raise ValueError('Frozen attempt')
 if not json.loads((compiled/'COMPLETE.json').read_text())['sram_passed']:raise ValueError('Complete joint banks have not passed SRAM')
 active=subprocess.run(['ssh','workstation','systemctl --user list-units --type=service --state=active,activating,deactivating --no-legend qwen38-single-*'],capture_output=True,text=True,check=True,timeout=20)
 if active.stdout.strip():raise ValueError('Live workstation owner: '+active.stdout)
 files={n:(compiled/'source'/n).read_bytes() for n in ['profiles.json','joint-bank-placement.json','joint-stage.json','mixer-setups.json']}
 files['compiled-source-manifest.json']=(compiled/'source-manifest.json').read_bytes()
 schedule=json.loads((e/'layer-native-schedule-002/layer-schedule.json').read_text());original=next(s for s in schedule['stages'] if s['id']=='layer_00')
 files['original-stage.json']=(json.dumps(original,separators=(',',':'))+'\n').encode()
 files['old-state-page-remap.json']=(e/'layer-mlp-hw-008/source/state-page-remap.json').read_bytes()
 baseline=e/'layer-mlp-reference-002';fixture=json.loads((baseline/'fixture.json').read_text());dispatch=json.loads((baseline/'dispatch.json').read_text())
 files['source-bank.json']=(json.dumps(dict(root=dispatch['remote'],hashes={n:fixture['hashes'][n] for n in ['banks.npy','bank-index.json']},source_fixture_sha256=hashlib.sha256((baseline/'fixture.json').read_bytes()).hexdigest()),indent=2)+'\n').encode()
 files['prepare.py']=(ROOT/'performance/reference/prepare_joint_banks.py').read_bytes()
 files['spatial/__init__.py']=b''
 for name0 in ['layer_schedule','joint_placement']:files['spatial/'+name0+'.py']=(ROOT/'performance/spatial'/(name0+'.py')).read_bytes()
 files['source_gate.py']=(ROOT/'runtime/source_gate.py').read_bytes();files['stager.py']=Path(__file__).read_bytes()
 files['execute.py']=rb'''import json,os,signal,subprocess,time
from pathlib import Path
from source_gate import verify
verify();started=time.monotonic()
try:
 with Path('prepare.log').open('x') as log:
  process=subprocess.Popen(['/usr/bin/python3','prepare.py'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   result=process.wait(timeout=300)
   if result:raise RuntimeError('Bank relocation exit '+str(result))
  finally:
   if process.poll() is None:
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=2)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=2)
 verify();Path('COMPLETE.json').write_text(json.dumps(dict(passed=True,physical=False,neural_execution=False,seconds=time.monotonic()-started,proof=json.loads(Path('bank-remap-proof.json').read_text())))+'\n')
except BaseException as error:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(error).__name__,message=str(error)))+'\n');raise
'''
 ast.parse(files['execute.py'])
 files['source-manifest.json']=(json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2)+'\n').encode()
 payload=io.BytesIO()
 with tarfile.open(fileobj=payload,mode='w') as archive:
  for n,b in files.items():
   item=tarfile.TarInfo(n);item.size=len(b);archive.addfile(item,io.BytesIO(b))
 remote='/srv/model-storage/qwen38-singlewse/runs/'+name
 subprocess.run(['ssh','workstation','mkdir '+shlex.quote(remote)+' && tar -xf - -C '+shlex.quote(remote)],input=payload.getvalue(),check=True,timeout=30)
 unit='qwen38-single-'+name
 cmd=['systemd-run','--user','--unit='+unit,'--property=MemoryMax=2G','--property=MemorySwapMax=0','--property=TasksMax=32',
      '--property=CPUQuota=100%','--property=AllowedCPUs=6','--property=RuntimeMaxSec=330','--property=TimeoutStopSec=5',
      '--property=KillMode=control-group','--property=LimitFSIZE=536870912','--property=LimitCORE=0','--working-directory='+remote,
      '--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=PYTHONDONTWRITEBYTECODE=1',
      '/usr/bin/taskset','--cpu-list','6','/usr/bin/flock','-n','/srv/cerebras-workstation/heavy.lock','/usr/bin/python3','execute.py']
 subprocess.run(['ssh','workstation',shlex.join(cmd)],check=True,timeout=20)
 out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
 (out/'dispatch.json').write_text(json.dumps(dict(remote=remote,unit=unit,physical=False,compile_attempt=compiled.name),indent=2)+'\n')
 print(json.dumps(dict(remote=remote,unit=unit,physical=False)))


if __name__=='__main__':main()
