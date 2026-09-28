"""Freeze independent full original mixer references in a bounded remote worker."""
import argparse,ast,hashlib,io,json,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
 p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--compile',default='017');a=p.parse_args()
 if any(len(v)!=3 or not v.isdigit() for v in (a.attempt,a.compile)):raise ValueError('Attempt')
 name='layer-mixer-reference-'+a.attempt;e=ROOT/'performance/evidence';out=e/name;compiled=e/('layer-mlp-compile-'+a.compile)
 if out.exists():raise ValueError('Frozen attempt')
 if not json.loads((compiled/'COMPLETE.json').read_text())['sram_passed']:raise ValueError('Unadmitted source stage')
 active=subprocess.run(['ssh','workstation','systemctl --user list-units --type=service --state=active,activating,deactivating --no-legend qwen38-single-*'],capture_output=True,text=True,check=True,timeout=20)
 if active.stdout.strip():raise ValueError('Live workstation owner: '+active.stdout)
 stage=json.loads((compiled/'source/joint-stage.json').read_text());region=next(r for r in stage['regions'] if r['role']=='mix')
 files={'region.json':(json.dumps(dict(matrices=region['matrices']),indent=2)+'\n').encode(),
        'compiled-source-manifest.json':(compiled/'source-manifest.json').read_bytes(),
        'tensors.json':(ROOT/'configs/tensors.json').read_bytes(),
        'prepare.py':(ROOT/'performance/reference/prepare_layer_mixer.py').read_bytes(),
        'reference/__init__.py':b'', 'stager.py':Path(__file__).read_bytes()}
 for n in ['mixer_oracle','mlp_oracle']:files['reference/'+n+'.py']=(ROOT/'performance/reference'/(n+'.py')).read_bytes()
 for n in ['weights','source_gate']:files[n+'.py']=(ROOT/'runtime'/(n+'.py')).read_bytes()
 files['execute.py']=rb'''import json,os,signal,subprocess,time
from pathlib import Path
from source_gate import verify
verify();started=time.monotonic()
try:
 with Path('prepare.log').open('x') as log:
  process=subprocess.Popen(['/usr/bin/python3','prepare.py'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   code=process.wait(timeout=300)
   if code:raise RuntimeError('Reference exit '+str(code))
  finally:
   if process.poll() is None:
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=2)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=2)
 verify();fixture=json.loads(Path('fixture.json').read_text());assert fixture['passed']
 Path('COMPLETE.json').write_text(json.dumps(dict(passed=True,physical=False,neural_device_execution=False,seconds=time.monotonic()-started,fixture=fixture))+'\n')
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
      '--property=KillMode=control-group','--property=LimitFSIZE=67108864','--property=LimitCORE=0','--working-directory='+remote,
      '--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=PYTHONDONTWRITEBYTECODE=1',
      '/usr/bin/taskset','--cpu-list','6','/usr/bin/flock','-n','/srv/cerebras-workstation/heavy.lock','/usr/bin/python3','execute.py']
 subprocess.run(['ssh','workstation',shlex.join(cmd)],check=True,timeout=20)
 out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
 (out/'dispatch.json').write_text(json.dumps(dict(remote=remote,unit=unit,physical=False,compile_attempt=compiled.name),indent=2)+'\n')
 print(json.dumps(dict(remote=remote,unit=unit,physical=False)))


if __name__=='__main__':main()
